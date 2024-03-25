
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

#include <fmt/format.h>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>

#include <opencv2/opencv.hpp>

#include <rerun.hpp>
#include <rerun/demo_utils.hpp>

#include <CLI/CLI.hpp>

#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/system_utils.h>
// #include <basalt/utils/vis_utils.h>
#include <basalt/utils/collection_adapter.hpp>
#include <basalt/utils/format.hpp>
#include <basalt/utils/time_utils.hpp>

using namespace rerun::demo;
using namespace basalt::literals;

// Visualization variables
std::unordered_map<int64_t, basalt::VioVisualizationData::Ptr> vis_map;

tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;
tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr>
    out_state_queue;

std::vector<int64_t> vio_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;
Eigen::aligned_vector<Sophus::SE3d> vio_T_w_i;

std::vector<int64_t> gt_t_ns;
Eigen::aligned_vector<Eigen::Vector3d> gt_t_w_i;

std::string marg_data_path;
size_t last_frame_processed = 0;

tbb::concurrent_unordered_map<int64_t, int, std::hash<int64_t>> timestamp_to_id;

std::mutex m;
std::condition_variable cv_image;
bool step_by_step = false;
size_t max_frames = 0;

std::atomic<bool> terminate = false;

// VIO variables
basalt::Calibration<double> calib;

basalt::VioDatasetPtr vio_dataset;
basalt::VioConfig vio_config;
basalt::OpticalFlowBase::Ptr opt_flow_ptr;
basalt::VioEstimatorBase::Ptr vio;

bool show_gui = true;
bool print_queue = false;
std::string cam_calib_path;
std::string dataset_path;
std::string dataset_type;
std::string config_path;
std::string result_path;
std::string trajectory_fmt;
bool trajectory_groundtruth;
int num_threads = 0;
bool use_imu = true;
bool use_double = false;

bool save_groundtruth = false;

void configure_cl() {
  int argc = 15;
  const char* argv[] = {
      "/home/yunsuk/workspace/basalt-mirror/build/basalt_vio_rerun",
      "--dataset-path",
      "/home/yunsuk/workspace/EUROC/V1_01_easy/",
      "--cam-calib",
      "/home/yunsuk/workspace/basalt-mirror/data/euroc_ds_calib.json",
      "--dataset-type",
      "euroc",
      "--config-path",
      "/home/yunsuk/workspace/basalt-mirror/data/euroc_config_vo.json",
      "--marg-data",
      "euroc_marg_data",
      "--use-double",
      "false",
      "--show-gui",
      "1"};

  CLI::App app{"App description"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--cam-calib", cam_calib_path,
                 "Ground-truth camera calibration used for simulation.")
      ->required();

  app.add_option("--dataset-path", dataset_path, "Path to dataset.")
      ->required();

  app.add_option("--dataset-type", dataset_type, "Dataset type <euroc, bag>.")
      ->required();

  app.add_option("--marg-data", marg_data_path,
                 "Path to folder where marginalization data will be stored.");

  app.add_option("--print-queue", print_queue, "Print queue.");
  app.add_option("--config-path", config_path, "Path to config file.");
  app.add_option("--result-path", result_path,
                 "Path to result file where the system will write RMSE ATE.");
  app.add_option("--num-threads", num_threads, "Number of threads.");
  app.add_option("--step-by-step", step_by_step, "Path to config file.");
  app.add_option("--save-trajectory", trajectory_fmt,
                 "Save trajectory. Supported formats <tum, euroc, kitti>");
  app.add_option("--save-groundtruth", trajectory_groundtruth,
                 "In addition to trajectory, save also ground turth");
  app.add_option("--use-imu", use_imu, "Use IMU.");
  app.add_option("--use-double", use_double, "Use double not float.");
  app.add_option(
      "--max-frames", max_frames,
      "Limit number of frames to process from dataset (0 means unlimited)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    std::cout << e.get_name() << std::endl;
    throw app.exit(e);
  }
};

void launch_rerun() {
  const auto rec = rerun::RecordingStream("rerun_example_cpp");
  // Try to spawn a new viewer instance.
  rec.spawn().exit_on_failure();

  // Create some data using the `grid` utility function.
  std::vector<rerun::Position3D> points =
      grid3d<rerun::Position3D, float>(-10.f, 10.f, 10);
  std::vector<rerun::Color> colors = grid3d<rerun::Color, uint8_t>(0, 255, 10);

  // Log the "my_points" entity with our data, using the `Points3D` archetype.
  rec.log("my_points",
          rerun::Points3D(points).with_colors(colors).with_radii({0.5f}));
}

void load_data(const std::string& calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib);
    std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras"
              << std::endl;

  } else {
    std::cerr << "could not load camera calibration " << calib_path
              << std::endl;
    std::abort();
  }
}

// Feed functions
void feed_images() {
  std::cout << "Started input_data thread " << std::endl;

  for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
    if (vio->finished || terminate || (max_frames > 0 && i >= max_frames)) {
      // stop loop early if we set a limit on number of frames to process
      break;
    }

    if (step_by_step) {
      std::unique_lock<std::mutex> lk(m);
      cv_image.wait(lk);
    }

    basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);

    data->t_ns = vio_dataset->get_image_timestamps()[i];
    data->img_data = vio_dataset->get_image_data(data->t_ns);

    timestamp_to_id[data->t_ns] = i;

    opt_flow_ptr->input_queue.push(data);
  }

  // Indicate the end of the sequence
  opt_flow_ptr->input_queue.push(nullptr);

  std::cout << "Finished input_data thread " << std::endl;
}

void feed_imu() {
  for (size_t i = 0; i < vio_dataset->get_gyro_data().size(); i++) {
    if (vio->finished || terminate) {
      break;
    }

    basalt::ImuData<double>::Ptr data(new basalt::ImuData<double>);
    data->t_ns = vio_dataset->get_gyro_data()[i].timestamp_ns;

    data->accel = vio_dataset->get_accel_data()[i].data;
    data->gyro = vio_dataset->get_gyro_data()[i].data;

    vio->imu_data_queue.push(data);
  }
  vio->imu_data_queue.push(nullptr);
}

int main() {
  // Create a new `RecordingStream` which sends data over TCP to the viewer
  // process.
  configure_cl();
  // launch_rerun();
  // global thread limit is in effect until global_control object is destroyed
  std::unique_ptr<tbb::global_control> tbb_global_control;
  if (num_threads > 0) {
    tbb_global_control = std::make_unique<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, num_threads);
  }
  // std::cout << config_path << std::endl;
  if (!config_path.empty()) {
    vio_config.load(config_path);

    if (vio_config.vio_enforce_realtime) {
      vio_config.vio_enforce_realtime = false;
      std::cout
          << "The option vio_config.vio_enforce_realtime was enabled, "
             "but it should only be used with the live executables (supply "
             "images at a constant framerate). This executable runs on the "
             "datasets and processes images as fast as it can, so the option "
             "will be disabled. "
          << std::endl;
    }
  }

  load_data(cam_calib_path);

  {
    basalt::DatasetIoInterfacePtr dataset_io =
        basalt::DatasetIoFactory::getDatasetIo(dataset_type);

    dataset_io->read(dataset_path);

    vio_dataset = dataset_io->get_data();

    // show_frame.Meta().range[1] = vio_dataset->get_image_timestamps().size() -
    // 1; show_frame.Meta().gui_changed = true;

    opt_flow_ptr =
        basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);

    for (size_t i = 0; i < vio_dataset->get_gt_pose_data().size(); i++) {
      gt_t_ns.push_back(vio_dataset->get_gt_timestamps()[i]);
      gt_t_w_i.push_back(vio_dataset->get_gt_pose_data()[i].translation());
    }
  }

  const int64_t start_t_ns = vio_dataset->get_image_timestamps().front();
  {
    vio = basalt::VioEstimatorFactory::getVioEstimator(
        vio_config, calib, basalt::constants::g, use_imu, use_double);
    vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    opt_flow_ptr->output_queue = &vio->vision_data_queue;
    if (show_gui) vio->out_vis_queue = &out_vis_queue;
    vio->out_state_queue = &out_state_queue;
  }

  basalt::MargDataSaver::Ptr marg_data_saver;

  if (!marg_data_path.empty()) {
    marg_data_saver.reset(new basalt::MargDataSaver(marg_data_path));
    vio->out_marg_queue = &marg_data_saver->in_marg_queue;

    // Save gt.
    {
      std::string p = marg_data_path + "/gt.cereal";
      std::ofstream os(p, std::ios::binary);

      {
        cereal::BinaryOutputArchive archive(os);
        archive(gt_t_ns);
        archive(gt_t_w_i);
      }
      os.close();
    }
  }

  // vio_data_log.Clear();

  std::thread t1(&feed_images);
  std::thread t2(&feed_imu);

  std::shared_ptr<std::thread> t3;

  if (show_gui)
    t3.reset(new std::thread([&]() {
      basalt::VioVisualizationData::Ptr data;

      while (true) {
        out_vis_queue.pop(data);

        if (data.get()) {
          vis_map[data->t_ns] = data;
        } else {
          break;
        }
      }

      std::cout << "Finished t3" << std::endl;
    }));

  std::thread t4([&]() {
    basalt::PoseVelBiasState<double>::Ptr data;

    while (true) {
      out_state_queue.pop(data);

      if (!data.get()) break;

      int64_t t_ns = data->t_ns;

      // std::cerr << "t_ns " << t_ns << std::endl;
      Sophus::SE3d T_w_i = data->T_w_i;
      Eigen::Vector3d vel_w_i = data->vel_w_i;
      Eigen::Vector3d bg = data->bias_gyro;
      Eigen::Vector3d ba = data->bias_accel;

      vio_t_ns.emplace_back(data->t_ns);
      vio_t_w_i.emplace_back(T_w_i.translation());
      vio_T_w_i.emplace_back(T_w_i);

      if (show_gui) {
        std::vector<float> vals;
        vals.push_back((t_ns - start_t_ns) * 1e-9);

        for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
        for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
        for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
        for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

        // vio_data_log.Log(vals);
      }
    }

    std::cout << "Finished t4" << std::endl;
  });
  std::shared_ptr<std::thread> t5;

  auto print_queue_fn = [&]() {
    std::cout << "opt_flow_ptr->input_queue "
              << opt_flow_ptr->input_queue.size()
              << " opt_flow_ptr->output_queue "
              << opt_flow_ptr->output_queue->size() << " out_state_queue "
              << out_state_queue.size() << " imu_data_queue "
              << vio->imu_data_queue.size() << std::endl;
  };

  if (print_queue) {
    t5.reset(new std::thread([&]() {
      while (!terminate) {
        print_queue_fn();
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }));
  }

  auto time_start = std::chrono::high_resolution_clock::now();

  // record if we close the GUI before VIO is finished.
  bool aborted = false;
  size_t show_frame = 0;

  const auto rec = rerun::RecordingStream("rerun_example_cpp");
  // Try to spawn a new viewer instance.
  auto result = rec.connect();
  if (result.code != rerun::ErrorCode::Ok) throw std::runtime_error("wtf");

  rec.log_timeless("world",
                   rerun::ViewCoordinates::RIGHT_HAND_Z_UP);  // Set an up-axis

  std::vector<rerun::Position3D> origins;
  std::vector<rerun::Vector3D> vectors;
  std::vector<rerun::Color> colors;

  origins.push_back({0, 0, 0});
  origins.push_back({0, 0, 0});
  origins.push_back({0, 0, 0});
  vectors.push_back({1.0f, 0.0f, 0.0f});
  vectors.push_back({0.0f, 1.0f, 0.0f});
  vectors.push_back({0.0f, 0.0f, 1.0f});
  colors.push_back({255, 0, 0});
  colors.push_back({0, 255, 0});
  colors.push_back({0, 0, 255});

  // auto z_arrow = rerun::Arrows3D::from_vectors({{0.0f, 0.0f, 1.0f}})
  //                    .with_origins({{0.0f, 0.0f, 0.0f}})
  //                    .with_colors({{0, 0, 255}});
  rec.log_timeless(
      "world/arrows",
      rerun::Arrows3D::from_vectors(vectors).with_origins(origins).with_colors(
          colors));

  uint64_t init_time = 0;
  if (show_gui) {
    while (true) {
      if (show_frame > vio_dataset->get_image_timestamps().size() - 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }

      if (!rec.is_enabled()) break;

      size_t frame_id = show_frame;
      int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];

      if (init_time == 0) init_time = t_ns;

      auto it = vis_map.find(t_ns);
      if (it != vis_map.end()) {
        show_frame++;
        auto T_w_c = it->second->states.back() * calib.T_i_c[0];
        // auto T_c_w = T_w_c.inverse();

        // std::cout << T_w_c.translation().transpose() << std::endl;
        Eigen::Vector3f camera_position  //= Eigen::Vector3f::Zero();
                                         // camera_position.z() = show_frame *
                                         // 0.001; = T_c_w.translation().z();
            = T_w_c.translation().cast<float>();
        Eigen::Matrix3f camera_orientation
            // camera_orientation << -0.382292, -0.00163536, -0.92404,
            // -0.00163536,
            //     0.999998, -0.00109321, 0.92404, 0.00109321, -0.382294;
            // = Eigen::Matrix3f::Identity();
            = T_w_c.so3().matrix().cast<float>();

        rec.set_time_nanos("time", t_ns - init_time);

        rec.log("world/camera",
                rerun::Transform3D(rerun::Vec3D(camera_position.data()),
                                   rerun::Mat3x3(camera_orientation.data())));

        rec.log("world/camera/image",
                rerun::Pinhole::from_focal_length_and_resolution({480.0, 480.0},
                                                                 {752.0, 480.0})
                    .with_camera_xyz(rerun::components::ViewCoordinates::RDF));
      }

      std::vector<basalt::ImageData> img_vec =
          vio_dataset->get_image_data(t_ns);

      size_t cam_id = 0;
      auto img = img_vec[cam_id].img.get();

      if (img) {
        cv::Mat image(img->h, img->w, CV_16UC1, img->ptr);
        rec.log("world/camera/image",
                rerun::Image(tensor_shape(image),
                             reinterpret_cast<const uint16_t*>(image.data)));
        // cv::imshow("wtf", image);
        // cv::waitKey();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  // wait first for vio to complete processing
  vio->maybe_join();

  // input threads will abort when vio is finished, but might be stuck in full
  // push to full queue, so drain queue now
  vio->drain_input_queues();

  // join input threads
  t1.join();
  t2.join();

  // std::cout << "Data input finished, terminate auxiliary threads.";
  terminate = true;

  // join other threads
  if (t3) t3->join();
  t4.join();
  if (t5) t5->join();

  // after joining all threads, print final queue sizes.
  if (print_queue) {
    std::cout << "Final queue sizes:" << std::endl;
    print_queue_fn();
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  const double duration_total =
      std::chrono::duration<double>(time_end - time_start).count();

  // TODO: remove this unconditional call (here for debugging);
  const double ate_rmse =
      basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i);
  vio->debug_finalize();
  std::cout << "Total runtime: {:.3f}s\n"_format(duration_total);

  {
    basalt::ExecutionStats stats;
    stats.add("exec_time_s", duration_total);
    stats.add("ate_rmse", ate_rmse);
    stats.add("ate_num_kfs", vio_t_w_i.size());
    stats.add("num_frames", vio_dataset->get_image_timestamps().size());

    {
      basalt::MemoryInfo mi;
      if (get_memory_info(mi)) {
        stats.add("resident_memory_peak", mi.resident_memory_peak);
      }
    }

    stats.save_json("stats_vio.json");
  }

  if (!aborted && !trajectory_fmt.empty()) {
    std::cout << "Saving trajectory..." << std::endl;

    // if (trajectory_fmt == "kitti") {
    //   kitti_fmt = true;
    //   euroc_fmt = false;
    //   tum_rgbd_fmt = false;
    // }
    // if (trajectory_fmt == "euroc") {
    //   euroc_fmt = true;
    //   kitti_fmt = false;
    //   tum_rgbd_fmt = false;
    // }
    // if (trajectory_fmt == "tum") {
    //   tum_rgbd_fmt = true;
    //   euroc_fmt = false;
    //   kitti_fmt = false;
    // }

    save_groundtruth = trajectory_groundtruth;

    // saveTrajectoryButton();
  }

  if (!aborted && !result_path.empty()) {
    double error = basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i);

    auto exec_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        time_end - time_start);

    std::ofstream os(result_path);
    {
      cereal::JSONOutputArchive ar(os);
      ar(cereal::make_nvp("rms_ate", error));
      ar(cereal::make_nvp("num_frames",
                          vio_dataset->get_image_timestamps().size()));
      ar(cereal::make_nvp("exec_time_ns", exec_time_ns.count()));
    }
    os.close();
  }

  return 0;
}