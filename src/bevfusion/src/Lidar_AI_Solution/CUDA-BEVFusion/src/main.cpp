/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cuda_runtime.h>
#include <string.h>

#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"

#include <visualization_msgs/Marker.h>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "sensor_msgs/PointCloud2.h" // 激光雷达的消息类型

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// 在代码中合适的位置添加命名空间声明
using namespace pcl;


cv::Mat zed_image;
visualization_msgs::Marker marker; // 创建立体矩形框消息对象
visualization_msgs::MarkerArray marker_array; // 创建一个MarkerArray对象
visualization_msgs::MarkerArray empty_marker_array;
ros::Publisher pub;

const char* data;
const char* model;
const char* precision;
std::shared_ptr<bevfusion::Core> core;

cudaStream_t stream;

// Load matrix to host
nv::Tensor camera2lidar;
nv::Tensor camera_intrinsics;
nv::Tensor lidar2image;
nv::Tensor img_aug_matrix;
// 点云数据 tensor
nv::Tensor lidar_point_cloud_tensor;



typedef unsigned short half;
static inline half __internal_float2half(const float f)
{
  unsigned int x;
  unsigned int u;
  unsigned int result;
  unsigned int sign;
  unsigned int remainder;
  (void)memcpy(&x, &f, sizeof(f));
  u = (x & 0x7fffffffU);
  sign = ((x >> 16U) & 0x8000U);
  
  if (u >= 0x7f800000U) {
      remainder = 0U;
      result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
  } else if (u > 0x477fefffU) {
      remainder = 0x80000000U;
      result = (sign | 0x7bffU);
  } else if (u >= 0x38800000U) {
      remainder = u << 19U;
      u -= 0x38000000U;
      result = (sign | (u >> 13U));
  } else if (u < 0x33000001U) {
      remainder = u;
      result = sign;
  } else {
      const unsigned int exponent = u >> 23U;
      const unsigned int shift = 0x7eU - exponent;
      unsigned int mantissa = (u & 0x7fffffU);
      mantissa |= 0x800000U;
      remainder = mantissa << (32U - shift);
      result = (sign | (mantissa >> shift));
      result &= 0x0000FFFFU;
  }

  unsigned short x_tmp = static_cast<unsigned short>(result);
  if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((x & 0x1U) != 0U))) {
      x_tmp++;
  }
  return x_tmp;
}



static std::vector<unsigned char*> load_images(const std::string& root) {
  const char* file_names[] = {"0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                              "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};

  std::vector<unsigned char*> images;
  for (int i = 0; i < 6; ++i) {
    char path[200];
    sprintf(path, "%s/%s", root.c_str(), file_names[i]);

    int width, height, channels;
    images.push_back(stbi_load(path, &width, &height, &channels, 0));
    // printf("Image info[%d]: %d x %d : %d\n", i, width, height, channels);
  }
  return images;
}

static void free_images(std::vector<unsigned char*>& images) {
  for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);

  images.clear();
}

static void visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points,
                      const std::vector<unsigned char*> images, const nv::Tensor& lidar2image, const std::string& save_path,
                      cudaStream_t stream) {
  std::vector<nv::Prediction> predictions(bboxes.size());
  memcpy(predictions.data(), bboxes.data(), bboxes.size() * sizeof(nv::Prediction));

  int padding = 300;
  int lidar_size = 1024;
  int content_width = lidar_size + padding * 3;
  int content_height = 1080;
  nv::SceneArtistParameter scene_artist_param;
  scene_artist_param.width = content_width;
  scene_artist_param.height = content_height;
  scene_artist_param.stride = scene_artist_param.width * 3;

  nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
  scene_device_image.memset(0x00, stream);

  scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
  auto scene = nv::create_scene_artist(scene_artist_param);

  nv::BEVArtistParameter bev_artist_param;
  bev_artist_param.image_width = content_width;
  bev_artist_param.image_height = content_height;
  bev_artist_param.rotate_x = 70.0f;
  bev_artist_param.norm_size = lidar_size * 0.5f;
  bev_artist_param.cx = content_width * 0.5f;
  bev_artist_param.cy = content_height * 0.5f;
  bev_artist_param.image_stride = scene_artist_param.stride;

  auto points = lidar_points.to_device();
  auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
  bev_visualizer->draw_lidar_points(points.ptr<nvtype::half>(), points.size(0));
  bev_visualizer->draw_prediction(predictions, false);
  bev_visualizer->draw_ego();
  bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

  nv::ImageArtistParameter image_artist_param;
  image_artist_param.num_camera = images.size();
  image_artist_param.image_width = 1600;
  image_artist_param.image_height = 900;
  image_artist_param.image_stride = image_artist_param.image_width * 3;
  image_artist_param.viewport_nx4x4.resize(images.size() * 4 * 4);
  memcpy(image_artist_param.viewport_nx4x4.data(), lidar2image.ptr<float>(),
         sizeof(float) * image_artist_param.viewport_nx4x4.size());

  int gap = 0;
  int camera_width = 500;
  int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
  int offset_cameras[][3] = {
      {-camera_width / 2, -content_height / 2 + gap, 0},
      {content_width / 2 - camera_width - gap, -content_height / 2 + camera_height / 2, 0},
      {-content_width / 2 + gap, -content_height / 2 + camera_height / 2, 0},
      {-camera_width / 2, +content_height / 2 - camera_height - gap, 1},
      {-content_width / 2 + gap, +content_height / 2 - camera_height - camera_height / 2, 0},
      {content_width / 2 - camera_width - gap, +content_height / 2 - camera_height - camera_height / 2, 1}};

  auto visualizer = nv::create_image_artist(image_artist_param);
  for (size_t icamera = 0; icamera < images.size(); ++icamera) {
    int ox = offset_cameras[icamera][0] + content_width / 2;
    int oy = offset_cameras[icamera][1] + content_height / 2;
    bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
    visualizer->draw_prediction(icamera, predictions, xflip);

    nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
    device_image.copy_from_host(images[icamera], stream);

    if (xflip) {
      auto clone = device_image.clone(stream);
      scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(),
                   device_image.size(1) * 3, stream);
      checkRuntime(cudaStreamSynchronize(stream));
    }
    visualizer->apply(device_image.ptr<unsigned char>(), stream);

    scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1),
                     device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
    checkRuntime(cudaStreamSynchronize(stream));
  }

  printf("Save to %s\n", save_path.c_str());
  stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3,
                 scene_device_image.to_host(stream).ptr(), 100);
}

std::shared_ptr<bevfusion::Core> create_core(const std::string& model, const std::string& precision) {

  printf("Create by %s, %s\n", model.c_str(), precision.c_str());
  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = 1600;
  normalization.image_height = 900;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = 6;
  normalization.resize_lim = 0.48f;
  normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

  float mean[3] = {0.485, 0.456, 0.406};
  float std[3] = {0.229, 0.224, 0.225};
  normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

  bevfusion::lidar::VoxelizationParameter voxelization;
  voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
  voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
  voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
  voxelization.grid_size =
      voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
  voxelization.max_points_per_voxel = 10;
  voxelization.max_points = 300000;
  voxelization.max_voxels = 160000;
  voxelization.num_feature = 5;

  bevfusion::lidar::SCNParameter scn;
  scn.voxelization = voxelization;
  scn.model = nv::format("/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/model/%s/lidar.backbone.xyz.onnx", model.c_str());
  scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

  if (precision == "int8") {
    scn.precision = bevfusion::lidar::Precision::Int8;
  } else {
    scn.precision = bevfusion::lidar::Precision::Float16;
  }

  bevfusion::camera::GeometryParameter geometry;
  geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
  geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
  geometry.image_width = 704;
  geometry.image_height = 256;
  geometry.feat_width = 88;
  geometry.feat_height = 32;
  geometry.num_camera = 6;
  geometry.geometry_dim = nvtype::Int3(360, 360, 80);

  bevfusion::head::transbbox::TransBBoxParameter transbbox;
  transbbox.out_size_factor = 8;
  transbbox.pc_range = {-54.0f, -54.0f};
  transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
  transbbox.post_center_range_end = {61.2, 61.2, 10.0};
  transbbox.voxel_size = {0.075, 0.075};
  transbbox.model = nv::format("/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/model/%s/build/head.bbox.plan", model.c_str());
  transbbox.confidence_threshold = 0.12f;
  transbbox.sorted_bboxes = true;

  bevfusion::CoreParameter param;
  param.camera_model = nv::format("/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/model/%s/build/camera.backbone.plan", model.c_str());
  param.normalize = normalization;
  param.lidar_scn = scn;
  param.geometry = geometry;
  param.transfusion = nv::format("/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/model/%s/build/fuser.plan", model.c_str());
  param.transbbox = transbbox;
  param.camera_vtransform = nv::format("/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/model/%s/build/camera.vtransform.plan", model.c_str());
  return bevfusion::create_core(param);
}


void Boxes2Txt(std::vector<bevfusion::head::transbbox::BoundingBox>  &boxes, std::string file_name, bool with_vel=false) {
  std::ofstream out_file;
  out_file.open(file_name, std::ios::out);
  if (out_file.is_open()) {
    for (const auto &box : boxes) {
      out_file << box.position.x << " ";
      out_file << box.position.y << " ";
      out_file << box.position.z << " ";
      out_file << box.size.l << " ";
      out_file << box.size.w << " ";
      out_file << box.size.h << " ";
      out_file << box.z_rotation << " ";
      if(with_vel){
        out_file << box.velocity.vx << " ";
        out_file << box.velocity.vy << " ";
      }
      out_file << box.score << " ";
      out_file << box.id << "\n";
    }
  }
  out_file.close();
  return;
};



void TestSample(){
  
  core->update(camera2lidar.ptr<float>(), camera_intrinsics.ptr<float>(), lidar2image.ptr<float>(), img_aug_matrix.ptr<float>(),
              stream);
  // core->free_excess_memory();

  // Load image and lidar to host
  auto images = load_images(data);


  // auto lidar_points = nv::Tensor::load(nv::format("%s/points.tensor", data), false);
  // lidar_points.print("lidar_point_cloud_tensor", 0, 5, 100);
  // run  lidar_point_cloud_tensor
  auto bboxes =
      core->forward((const unsigned char**)images.data(), lidar_point_cloud_tensor.ptr<nvtype::half>(), lidar_point_cloud_tensor.size(0), stream);

  //输出框的信息
  Boxes2Txt(bboxes, "/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/result/1.txt", false);
  // visualize and save to jpg
  // visualize(bboxes, lidar_point_cloud_tensor, images, lidar2image, "/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/build/cuda-bevfusion.jpg", stream);

  // destroy memory
  free_images(images);
  // checkRuntime(cudaStreamDestroy(stream));

  // marker 
  if(bboxes.empty()){
    printf("lidarbox = 0  \n");

  }else{
    for (size_t i = 0; i < bboxes.size(); ++i) {
      if(bboxes[i].score > 0.15 && bboxes[i].id == 8){
        marker.header.frame_id = "rslidar"; // 设置Marker的坐标系

        marker.header.stamp = ros::Time::now();
        marker.ns = "basic_shapes";
        marker.id = i;
        marker.lifetime = ros::Duration(1);
        marker.type = visualization_msgs::Marker::CUBE; // 设置Marker类型为立方体
        marker.action = visualization_msgs::Marker::ADD;

        // 设置长方体的尺寸
        marker.scale.x = bboxes[i].size.l; // 长
        marker.scale.y = bboxes[i].size.w; // 宽
        marker.scale.z = bboxes[i].size.h; // 高

        // 设置Marker的颜色
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.5; // 半透明

        marker.pose.position.x = bboxes[i].position.x; // 设置长方体的位置
        marker.pose.position.y = bboxes[i].position.y;
        marker.pose.position.z = bboxes[i].position.z;

        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        
        marker_array.markers.push_back(marker); // 将标记添加到MarkerArray中

      }else{
        // pub.publish(empty_marker_array); // 发布消息到话题上
      }
    }

  }
  pub.publish(marker_array); // 发布消息到话题上
  marker_array.markers.clear();
  bboxes.clear();
 




}




void imageCb(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }    
  cv::Mat zed_image_copy;
  cv_ptr->image.copyTo(zed_image_copy);
  cv::Size targetSize(1600, 900);
  cv::resize(zed_image_copy, zed_image, targetSize, cv::INTER_LINEAR); // 可以选择不同的插值方法
  
  cv::imshow("zed_image", zed_image_copy);
  cv::waitKey(1);
  cv::imwrite("/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/example-data/0-FRONT.jpg", zed_image);

  TestSample();
}

void lidarCb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr ROI_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *ROI_cloud);
  std::cout << "ROI_cloud->points.size()   " << ROI_cloud->points.size() << std::endl;

  int last_j = 0;
  for (int i = 0, j = 0; i < ROI_cloud->points.size(); i++)
  {
    if(std::isnan(ROI_cloud->points[i].x) || std::isnan(ROI_cloud->points[i].y) || std::isnan(ROI_cloud->points[i].z)){

    }else{
      j++;
    }
    last_j = j;
  }
  printf(" last_j = %d \n", last_j);
  half *points = new half[last_j * 5];
  for (int i = 0, j = 0; i < ROI_cloud->points.size(); i++)
  {
    if(std::isnan(ROI_cloud->points[i].x) || std::isnan(ROI_cloud->points[i].y) || std::isnan(ROI_cloud->points[i].z)){
      
    }else{
      points[j * 5 + 0] = __internal_float2half(ROI_cloud->points[i].x);
      points[j * 5 + 1] = __internal_float2half(ROI_cloud->points[i].y);
      points[j * 5 + 2] = __internal_float2half(ROI_cloud->points[i].z);
      points[j * 5 + 3] = __internal_float2half(ROI_cloud->points[i].intensity);
      points[j * 5 + 4] = __internal_float2half(0);
      j++;
    }
  }

  std::vector<int32_t> shape{last_j, 5};
  // Tensor Tensor::from_data_reference(void *data, vector<int32_t> shape, DataType dtype, bool device)
  lidar_point_cloud_tensor = nv::Tensor::from_data_reference(points, shape, nv::DataType::Float16, false);
  // lidar_point_cloud_tensor.print("lidar_point_cloud_tensor", 0, 5, last_j);
}


int main(int argc, char** argv) {

  data      = "/home/orin_uestc_1/bevformer_ws/src/bevfusion/src/Lidar_AI_Solution/CUDA-BEVFusion/example-data";
  model     = "resnet50int8";
  precision = "int8";

  if (argc > 1) data      = argv[1];
  if (argc > 2) model     = argv[2];
  if (argc > 3) precision = argv[3];

  core = create_core(model, precision);
  if (core == nullptr) {
    printf("Core has been failed.\n");
    return -1;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);
 
  core->print();
  core->set_timer(true);

  // Load matrix to host
  camera2lidar = nv::Tensor::load(nv::format("%s/camera2lidar.tensor", data), false);
  camera_intrinsics = nv::Tensor::load(nv::format("%s/camera_intrinsics.tensor", data), false);
  lidar2image = nv::Tensor::load(nv::format("%s/lidar2image.tensor", data), false);
  img_aug_matrix = nv::Tensor::load(nv::format("%s/img_aug_matrix.tensor", data), false);
  printf("tensor info - >\n");
  camera2lidar.print("Tensor", 0, 4, 4);
  printf("\n");

  // float data_1[96] = {
  //     0.99993889, 0.00386537, 0.00707301, 0.01224191,
  //     0.00694007, 0.01901997, -0.99959004, -0.32533256,
  //     -0.00313168, 0.99962702, 0.01897099, -0.75885541,
  //     0.0, 0.0, 0.0, 1.0,
  // };
  // void * data_2 = data_1;
  // std::vector<int64_t> shape = {1, 6, 4, 4};
  // lidar2image = nv::Tensor::from_data_reference((void *)data_2, shape, nv::DataType::Float32, false);
  camera_intrinsics.print("Tensor", 0, 4, 4);
  printf("\n");
  lidar2image.print("Tensor", 0, 4, 4);
  printf("\n");  
  img_aug_matrix.print("Tensor", 0, 4, 4);
  printf("tensor info end.\n");

  // 初始化 ROS 节点
  ros::init(argc, argv, "image_subscriber");

  // 创建 ROS 句柄
  ros::NodeHandle nh;
  image_transport::Subscriber image_sub_;
  ros::Subscriber lidar_sub_;
  image_transport::ImageTransport it(nh);
  pub = nh.advertise<visualization_msgs::MarkerArray>("markerarray_topic", 10); // 定义发布器
  ros::Rate loop_rate(20); // 发布频率为10Hz
  // 创建订阅器，订阅图像消息
  image_sub_ = it.subscribe("/zed2i/zed_node/rgb/image_rect_color", 1, &imageCb);
  lidar_sub_ = nh.subscribe<sensor_msgs::PointCloud2>("rslidar_points", 1, &lidarCb);

  // 进入 ROS 循环
  ros::spin();

  return 0;

}