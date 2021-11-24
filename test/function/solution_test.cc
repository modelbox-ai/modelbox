/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <sys/stat.h>

#include <atomic>
#include <cstdio>
#include <fstream>
#include <functional>
#include <future>
#include <thread>

#include "modelbox/base/log.h"
#include "modelbox/buffer.h"
#include "modelbox/flow.h"
#include "modelbox/graph.h"
#include "modelbox/node.h"
#include "engine/scheduler/flow_scheduler.h"
#include "flowunit_mockflowunit/flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/mock/minimodelbox/mockflow.h"

namespace modelbox {
using ::testing::Sequence;
class SolutionTest : public testing::Test {
 public:
  SolutionTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

static bool SkipJudge(const std::string &local_file_path) {
  struct stat statbuf;
  if (stat(local_file_path.c_str(), &statbuf) == -1) {
    MBLOG_ERROR << "failed to load " << local_file_path;
    return true;
  }
  return false;
}

static std::string GetModelPath(const std::string &toml_path) {
  auto conf_builder = std::make_shared<ConfigurationBuilder>();
  std::shared_ptr<Configuration> toml_config = conf_builder->Build(toml_path);
  auto model_path = toml_config->GetString("base.entry");
  return model_path;
}

static void TestRunGraph(const std::string &toml_content) {
  std::string config_file_path =
      std::string(TEST_DATA_DIR) + "/solution_test.toml";
  struct stat buffer;
  if (stat(config_file_path.c_str(), &buffer) == 0) {
    remove(config_file_path.c_str());
  }
  std::ofstream solution_test_toml(config_file_path);
  EXPECT_TRUE(solution_test_toml.is_open());
  solution_test_toml.write(toml_content.data(), toml_content.size());
  solution_test_toml.flush();
  solution_test_toml.close();
  Defer {
    auto rmret = remove(config_file_path.c_str());
    EXPECT_EQ(rmret, 0);
  };

  auto flow = std::make_shared<Flow>();
  auto ret = flow->Init(config_file_path);
  EXPECT_EQ(ret, STATUS_OK);

  ret = flow->Build();
  EXPECT_EQ(ret, STATUS_OK);

  flow->RunAsync();
  flow->Wait(0);
}

TEST_F(SolutionTest, LPDetection) {
  std::string video_file =
      std::string(TEST_SOLUTION_VIDEO_DIR) + std::string("/test_video.mp4");
  if (SkipJudge(video_file) == true) {
    GTEST_SKIP();
  };

  std::string car_toml = std::string(TEST_SOLUTION_DRIVERS_DIR) +
                         std::string("/car_inference.toml");
  std::string car_model_path = GetModelPath(car_toml);
  if (SkipJudge(car_model_path) == true) {
    GTEST_SKIP();
  };

  std::string lp_toml = std::string(TEST_SOLUTION_DRIVERS_DIR) +
                        std::string("/lp_inference.toml");
  std::string lp_model_path = GetModelPath(lp_toml);
  if (SkipJudge(lp_model_path) == true) {
    GTEST_SKIP();
  };

  MBLOG_INFO << "lp detection get in." << std::endl;
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string test_solution_dir = TEST_SOLUTION_DRIVERS_DIR;
  std::string toml_content = R"(
    [log]
    level = "INFO"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\", \"" +
                             test_solution_dir + "\"]\n    " +
                             R"([graph]
    graphconf = """digraph solution_test {
                video_input[type=flowunit, flowunit=video_input, device=cpu, deviceid=0, label="<out_video_url>", source_url="/opt/modelbox/solution/video/test_video.mp4"]                                           
                videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0, label="<in_video_url> | <out_video_packet>"]
                videodecoder[type=flowunit, flowunit=video_decoder, device=cpu, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=rgb, queue_size = 16, batch_size=5]
                frame_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, label="<in_image> | <out_image>", image_width=800, image_height=480, method="inter_nearest", batch_size=5, queue_size = 16]
                car_color_transpose[type=flowunit, flowunit=packed_planar_transpose, device=cpu, deviceid=0, label="<in_image> | <out_image>", batch_size = 5, queue_size = 16]
                car_normalize[type=flowunit, flowunit=normalize, device=cpu, deviceid=0, label="<input> | <output>", normalize="0.003921568627451, 0.003921568627451, 0.003921568627451", queue_size = 16, batch_size = 5]
                car_inference[type=flowunit, flowunit=car_inference, device=cuda, deviceid=0, label="<data> | <layer15-conv> | <layer22-conv>", queue_size = 16, batch_size = 5]
                car_yolobox[type=flowunit, flowunit=car_yolobox, device=cpu, deviceid=0, label="<layer15-conv> | <layer22-conv> | <Out_1>", image_width=1920, image_height=1080, queue_size = 16, batch_size = 5]
                expand_bbox_img[type=flowunit, flowunit=expand_bbox_img, device=cpu, deviceid=0, label="<In_img> | <In_bbox> | <Out_img> | <Out_bbox>"]

                car_condition[type=flowunit, flowunit=car_condition, device=cpu, deviceid=0, label="<In_img> | <In_bbox> | <Out_true> | <Out_false>", batch_size = 1, queue_size = 16]
                split_img_bbox[type=flowunit, flowunit=split_img_bbox, device=cpu, deviceid=0, label="<In_true> | <Out_img> | <Out_bbox>", batch_size = 5, queue_size = 16]            
                
                car_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, label="<in_image> | <Out_1>", image_width=416, image_height=416, method="inter_nearest", batch_size=5, queue_size = 16]
                lp_color_transpose[type=flowunit, flowunit=packed_planar_transpose, device=cpu, deviceid=0, label="<in_image> | <out_image>", batch_size = 5, queue_size = 16]
                lp_normalize[type=flowunit, flowunit=normalize, device=cpu, deviceid=0, label="<input> | <output>", normalize="0.003921568627451, 0.003921568627451, 0.003921568627451", queue_size = 16, batch_size = 5]
                lp_inference[type=flowunit, flowunit=lp_inference, device=cuda, deviceid=0, label="<data> | <layer11-conv>", queue_size = 16, batch_size = 5]
                lp_yolobox[type=flowunit, flowunit=lp_yolobox, device=cpu, deviceid=0, label="<layer11-conv> | <car_bboxes> | <Out_1>", image_width=1920, image_height=1080, queue_size = 16, batch_size = 5]
                collapse_bbox[type=flowunit, flowunit=collapse_bbox, device=cpu, deviceid=0, label="<input> | <output>"]
                draw_bbox[type=flowunit, flowunit=draw_bbox, device=cpu, deviceid=0, label="<in_image> | <in_region> | <out_image>", queue_size = 16, batch_size = 5]
                videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, deviceid=0, label="<in_video_frame>", queue_size=16, default_dest_url="rtsp://172.22.115.16/test", encoder="mpeg4"]
                
                video_input:out_video_url -> videodemuxer:in_video_url
                videodemuxer:out_video_packet -> videodecoder:in_video_packet
                videodecoder:out_video_frame -> frame_resize:in_image
                frame_resize: out_image -> car_color_transpose: in_image
                car_color_transpose: out_image -> car_normalize: in_data
                car_normalize: out_data -> car_inference:data
                car_inference: "layer15-conv" -> car_yolobox: "layer15-conv"
                car_inference: "layer22-conv" -> car_yolobox: "layer22-conv"
                car_yolobox: Out_1 -> expand_bbox_img: In_bbox
                videodecoder:out_video_frame -> expand_bbox_img: In_img
                expand_bbox_img: Out_img -> car_condition: In_img
                expand_bbox_img: Out_bbox -> car_condition: In_bbox

                car_condition: Out_true -> split_img_bbox: In_true
                car_condition: Out_false -> collapse_bbox: input

                split_img_bbox: Out_img -> car_resize: in_image
                split_img_bbox: Out_bbox -> lp_yolobox: car_bboxes

                car_resize: out_image  -> lp_color_transpose: in_image
                lp_color_transpose: out_image  ->  lp_normalize: in_data
                lp_normalize: out_data -> lp_inference: data
                lp_inference: "layer11-conv" -> lp_yolobox: "layer11-conv"
                lp_yolobox: Out_1 -> collapse_bbox: input
                collapse_bbox: output -> draw_bbox: in_region
                videodecoder:out_video_frame -> draw_bbox: in_image 
                draw_bbox: out_image -> videoencoder: in_video_frame   
                }"""
    format = "graphviz"
  )";

  // run graph
  TestRunGraph(toml_content);
}

TEST_F(SolutionTest, CarDetection) {
  std::string video_file =
      std::string(TEST_SOLUTION_VIDEO_DIR) + std::string("/test_video.mp4");
  if (SkipJudge(video_file) == true) {
    GTEST_SKIP();
  };

  std::string car_toml = std::string(TEST_SOLUTION_DRIVERS_DIR) +
                         std::string("/car_inference.toml");
  std::string car_model_path = GetModelPath(car_toml);
  if (SkipJudge(car_model_path) == true) {
    GTEST_SKIP();
  };

  MBLOG_INFO << "car detection get in." << std::endl;
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string test_solution_dir = TEST_SOLUTION_DRIVERS_DIR;
  std::string toml_content = R"(
    [log]
    level = "INFO"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\", \"" +
                             test_solution_dir + "\"]\n    " +
                             R"([graph]
    graphconf = """digraph solution_test {
                video_input[type=flowunit, flowunit=video_input, device=cpu, deviceid=0, label="<out_video_url>", source_url="/opt/modelbox/solution/video/test_video.mp4"]                                           
                videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0, label="<in_video_url> | <out_video_packet>"]
                videodecoder[type=flowunit, flowunit=video_decoder, device=cpu, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=rgb, queue_size = 16, batch_size=5]
                frame_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, label="<in_image> | <Out_1>", image_width=800, image_height=480, method="inter_nearest", batch_size=5, queue_size = 16]
                car_color_transpose[type=flowunit, flowunit=packed_planar_transpose, device=cpu, deviceid=0, label="<in_image> | <out_image>", batch_size = 5, queue_size = 16]
                car_normalize[type=flowunit, flowunit=normalize, device=cpu, deviceid=0, label="<in_data> | <out_data>", normalize="0.003921568627451, 0.003921568627451, 0.003921568627451", queue_size = 16, batch_size = 5]
                car_inference[type=flowunit, flowunit=car_inference, device=cuda, deviceid=0, label="<data> | <layer15-conv> | <layer22-conv>", queue_size = 16, batch_size = 5]
                car_yolobox[type=flowunit, flowunit=car_yolobox, device=cpu, deviceid=0, label="<layer15-conv> | <layer22-conv> | <Out_1>", image_width=1920, image_height=1080, queue_size = 16, batch_size = 5]
                draw_bbox[type=flowunit, flowunit=draw_bbox, device=cpu, deviceid=0, label="<in_image> | <in_region> | <out_image>", queue_size = 16, batch_size = 5]
                videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, deviceid=0, label="<in_video_frame>", queue_size=16, default_dest_url="rtsp://localhost/test", encoder="mpeg4"]
                
                video_input:out_video_url -> videodemuxer:in_video_url
                videodemuxer:out_video_packet -> videodecoder:in_video_packet
                videodecoder:out_video_frame -> frame_resize:in_image
                frame_resize: out_image -> car_color_transpose: in_image
                car_color_transpose: out_image -> car_normalize: in_data
                car_normalize: out_data -> car_inference:data
                car_inference: "layer15-conv" -> car_yolobox: "layer15-conv"
                car_inference: "layer22-conv" -> car_yolobox: "layer22-conv"
                car_yolobox: Out_1 -> draw_bbox: in_region
                videodecoder:out_video_frame -> draw_bbox: in_image
                draw_bbox: out_image -> videoencoder: in_video_frame   
                }"""
    format = "graphviz"
  )";

  TestRunGraph(toml_content);
}

TEST_F(SolutionTest, YOLOv3) {
  std::string video_file =
      std::string(TEST_SOLUTION_VIDEO_DIR) + std::string("/test_video.mp4");
  if (SkipJudge(video_file) == true) {
    GTEST_SKIP();
  };

  std::string yolov3_toml = std::string(TEST_SOLUTION_DRIVERS_DIR) +
                            std::string("/yolov3_inference.toml");
  std::string yolov3_model_path = GetModelPath(yolov3_toml);
  if (SkipJudge(yolov3_model_path) == true) {
    GTEST_SKIP();
  };

  MBLOG_INFO << "YOLOv3(coco) get in." << std::endl;
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string test_solution_dir = TEST_SOLUTION_DRIVERS_DIR;
  std::string toml_content = R"(
    [log]
    level = "INFO"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\", \"" +
                             test_solution_dir + "\"]\n    " +
                             R"([graph]
    graphconf = """digraph solution_test {
                video_input[type=flowunit, flowunit=video_input, device=cpu, deviceid=0, source_url="/opt/modelbox/solution/video/test_video.mp4"]                                                                                 
                videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0]
                videodecoder[type=flowunit, flowunit=video_decoder, device=cpu, deviceid=0, pix_fmt=rgb, queue_size = 16, batch_size=5]
                frame_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, image_width=608, image_height=608, method="inter_nearest", batch_size=5, queue_size = 16]
                color_transpose[type=flowunit, flowunit=packed_planar_transpose, device=cpu, deviceid=0, batch = 5, queue_size = 16]
                normalize[type=flowunit, flowunit=normalize, device=cpu, deviceid=0, normalize="0.003921568627451, 0.003921568627451, 0.003921568627451", queue_size = 16, batch_size = 5]
                yolov3_inference[type=flowunit, flowunit=yolov3_inference, device=cuda, deviceid=0, queue_size = 16, batch_size = 5]
                YOLOv3_post[type=flowunit, flowunit=YOLOv3_post, device=cpu, deviceid=0, image_width=1920, image_height=1080, queue_size = 16, batch_size = 5]
                draw_bbox[type=flowunit, flowunit=draw_bbox, device=cpu, deviceid=0, queue_size = 16, batch_size = 5]
                videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, deviceid=0, queue_size=16, default_dest_url="rtsp://localhost/test", encoder="mpeg4"]

                video_input:out_video_url -> videodemuxer:in_video_url
                videodemuxer:out_video_packet -> videodecoder:in_video_packet
                videodecoder:out_video_frame -> frame_resize:in_image
                frame_resize: out_image -> color_transpose: in_image
                color_transpose: out_image -> normalize: in_data
                normalize: out_data -> yolov3_inference:data
                yolov3_inference: "layer82-conv" -> YOLOv3_post: "layer82-conv"
                yolov3_inference: "layer94-conv" -> YOLOv3_post: "layer94-conv"
                yolov3_inference: "layer106-conv" -> YOLOv3_post: "layer106-conv"
                yolov3: Out_1 -> draw_bbox: in_region
                videodecoder:out_video_frame -> draw_bbox: in_image
                draw_bbox: out_image -> videoencoder: in_video_frame      
                }"""
    format = "graphviz"
  )";

  TestRunGraph(toml_content);
}

TEST_F(SolutionTest, FaceDetection) {
  std::string video_file =
      std::string(TEST_SOLUTION_VIDEO_DIR) + std::string("/face_test.mp4");
  if (SkipJudge(video_file) == true) {
    GTEST_SKIP();
  };

  std::string face_toml = std::string(TEST_SOLUTION_DRIVERS_DIR) +
                          std::string("/face_inference.toml");
  std::string face_model_path = GetModelPath(face_toml);
  if (SkipJudge(face_model_path) == true) {
    GTEST_SKIP();
  };

  MBLOG_INFO << "face detection get in." << std::endl;
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string test_solution_dir = TEST_SOLUTION_DRIVERS_DIR;
  std::string toml_content = R"(
    [log]
    level = "INFO"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\", \"" +
                             test_solution_dir + "\"]\n    " +
                             R"([graph]
    graphconf = '''digraph demo {
            video_input[type=flowunit, flowunit=video_input, device=cpu, deviceid=0, label="<out_video_url>", source_url="/opt/modelbox/solution/video/face_test.mp4"]
            videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0, label="<in_video_url> | <out_video_packet>"]
            videodecoder[type=flowunit, flowunit=video_decoder, device=cpu, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=rgb, queue_size=16, batch_size=5]
            face_preprocess[type=flowunit, flowunit=face_preprocess, device=cpu, deviceid=0, label="<In_1> | <Out_1>", width=640, height=352, resize_type=1, method="inter_nearest", batch_size=5, queue_size=16]
            face_color_transpose[type=flowunit, flowunit=face_color_transpose, device=cpu, deviceid=0, label="<in_image> | <out_image>", batch_size=5, queue_size=16]
            face_inference[type=flowunit, flowunit=face_inference, device=cuda, deviceid=0, label="<blob1> | <sigmoid_blob1> | <conv_blob60> | <conv_blob62> | <conv_blob64>", queue_size=16, batch_size=5]
            face_center[type=flowunit, flowunit=face_center, device=cpu, deviceid=0, label="<sigmoid_blob1> | <conv_blob60> | <conv_blob62> | <conv_blob64> | <Out_1> | <Out_2>", image_width=2560, image_height=1440, input_width=640, input_height=352, queue_size=16, batch_size=5]

            face_alignment[type=flowunit, flowunit=face_alignment, device=cpu, deviceid=0, label="<In_img> | <In_kps> | <Aligned_img>", net_width=224, net_height=224, batch_size=5, queue_size=16]
            face_expand[type=flowunit, flowunit=face_expand, device=cpu, deviceid=0, label="<In_img> | <Out_img>"]
            face_condition[type=flowunit, flowunit=face_condition, device=cpu, deviceid=0, label="<In_img> | <Out_true> | <Out_false>", queue_size=16, batch_size=1]
            expression_inference[type=flowunit, flowunit=expression_inference, device=cuda, deviceid=0, label="<blob1> | <fc_blob1>", queue_size=16, batch_size=5]
            expression_process[type=flowunit, flowunit=face_mobilev2, device=cpu, deviceid=0, label="<fc_blob1> | <Out_1>", queue_size=16, batch_size=5]
            face_collapse[type=flowunit, flowunit=face_collapse, device=cpu, deviceid=0, label="<In_label> | <Out_label>"]
            face_draw[type=flowunit, flowunit=face_draw, device=cpu, deviceid=0, label="<In_1> | <In_2> | <In_3> | <Out_1>", method=max, queue_size=16, batch_size=5]
            videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, deviceid=0, label="<in_video_frame>", queue_size=16, default_dest_url="rtsp://172.22.115.16/youxujia_test", encoder="mpeg4"]

            video_input:out_video_url -> videodemuxer:in_video_url
            videodemuxer:out_video_packet -> videodecoder:in_video_packet
            videodecoder:out_video_frame -> face_preprocess:In_1
            face_preprocess: Out_1 -> face_color_transpose: in_image
            face_color_transpose: out_image -> face_inference:blob1
            face_inference: sigmoid_blob1 -> face_center: sigmoid_blob1
            face_inference: conv_blob60 -> face_center: conv_blob60
            face_inference: conv_blob62 -> face_center: conv_blob62
            face_inference: conv_blob64 -> face_center: conv_blob64
            face_center: Out_1 -> face_draw: In_1
            face_center:Out_2 -> face_alignment:In_kps

            videodecoder:out_video_frame -> face_alignment:In_img
            face_alignment:Aligned_img -> face_expand:In_img
            face_expand:Out_img -> face_condition : In_img
            face_condition:Out_false -> face_collapse:In_label
            face_condition:Out_true -> expression_inference : blob1
            expression_inference:fc_blob1 -> expression_process:fc_blob1
            expression_process:Out_1 -> face_collapse:In_label
            face_collapse:Out_label -> face_draw:In_3
            videodecoder:out_video_frame -> face_draw: In_2
            face_draw: Out_1 -> videoencoder: in_video_frame
            }'''
    format = "graphviz"
  )";

  TestRunGraph(toml_content);
}

TEST_F(SolutionTest, PedestrianTracking) {
  std::string video_file =
      std::string("/opt/modelbox/solution/video/ppt_1080p.mp4");
  if (SkipJudge(video_file) == true) {
    GTEST_SKIP();
  };

  std::string pedestrian_toml =
      std::string(TEST_SOLUTION_DRIVERS_DIR) +
      std::string("/pedestrian_detect_inference.toml");
  std::string pedestrian_model_path = GetModelPath(pedestrian_toml);
  if (SkipJudge(pedestrian_model_path) == true) {
    GTEST_SKIP();
  };

  std::string reid_toml = std::string(TEST_SOLUTION_DRIVERS_DIR) +
                          std::string("/pedestrian_reid_inference.toml");
  std::string reid_model_path = GetModelPath(reid_toml);
  if (SkipJudge(reid_model_path) == true) {
    GTEST_SKIP();
  };

  MBLOG_INFO << "pedestrian tracking get in." << std::endl;
  const std::string test_lib_dir = TEST_DRIVER_DIR;
  const std::string test_solution_dir = TEST_SOLUTION_DRIVERS_DIR;
  std::string toml_content = R"(
    [log]
    level = "DEBUG"
    [driver]
    skip-default=true
    dir=[")" + test_lib_dir + "\", \"" +
                             test_solution_dir + "\"]\n    " +
                             R"([graph]
    graphconf = """digraph solution_test {
              video_input[type=flowunit, flowunit=video_input, device=cpu, deviceid=0, label="<out_video_url>", source_url="/opt/modelbox/solution/video/ppt_1080p.mp4"]
              videodemuxer[type=flowunit, flowunit=video_demuxer, device=cpu, deviceid=0, label="<in_video_url> | <out_video_packet>"]
              videodecoder[type=flowunit, flowunit=video_decoder, device=cpu, deviceid=0, label="<in_video_packet> | <out_video_frame>", pix_fmt=rgb, queue_size = 16]
              skip_frame[type=flowunit, flowunit=skip_frame, device=cpu, deviceid=0, label="<input_frame> | <output_frame>", process_frame_per_second=5, queue_size = 16]
              fullImg_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, label="<in_image> | <out_image>", width=800, height=480, method="inter_nearest", batch_size=5, queue_size = 16]
              fullImg_color_transpose[type=flowunit, flowunit=packed_planar_transpose, device=cpu, deviceid=0, label="<in_image> | <out_image>", queue_size = 16]
              fullImg_normalize[type=flowunit, flowunit=normalize, device=cpu, deviceid=0, label="<in_data> | <out_data>", normalize="0.003921568627451, 0.003921568627451, 0.003921568627451", queue_size = 16, batch_size = 5]
              pedestrian_detect_inference[type=flowunit, flowunit=pedestrian_detect_inference, device=cuda, deviceid=0, label="<data> | <layer82-conv> | <layer94-conv> | <layer106-conv>", queue_size = 16, batch_size = 5]
              pedestrian_yolov3_post[type=flowunit, flowunit=pedestrian_yolov3_post, device=cpu, deviceid=0, label="<layer82-conv> | <layer94-conv> | <layer106-conv> | <Out_1>", queue_size = 16, batch_size = 5]

              video_input:out_video_url -> videodemuxer:in_video_url
              videodemuxer:out_video_packet -> videodecoder:in_video_packet
              videodecoder:out_video_frame -> skip_frame:input_frame
              skip_frame:output_frame -> fullImg_resize:in_image
              fullImg_resize: out_image -> fullImg_color_transpose: in_image
              fullImg_color_transpose: out_image -> fullImg_normalize: in_data
              fullImg_normalize: out_data -> pedestrian_detect_inference:data
              pedestrian_detect_inference: "layer82-conv" -> pedestrian_yolov3_post: "layer82-conv"
              pedestrian_detect_inference: "layer94-conv" -> pedestrian_yolov3_post: "layer94-conv"
              pedestrian_detect_inference: "layer106-conv" -> pedestrian_yolov3_post: "layer106-conv"

              expand_bbox_img[type=flowunit, flowunit=pedestrian_expand_bbox_img, device=cpu, deviceid=0, label="<In_img> | <In_bbox> | <Out_img> | <Out_bbox>"]
              has_bbox_condition[type=flowunit, flowunit=has_bbox_condition, device=cpu, deviceid=0, label="<In_img> | <In_bbox> | <Out_true> | <Out_false>", batch_size = 1, queue_size = 16]
              split_img_bbox[type=flowunit, flowunit=split_img_bbox, device=cpu, deviceid=0, label="<In_true> | <Out_img> | <Out_bbox>", batch_size = 8, queue_size = 16]
              cropImg_resize[type=flowunit, flowunit=resize, device=cpu, deviceid=0, label="<in_image> | <out_image>", width=128, height=256, method="inter_nearest", batch_size=8, queue_size = 16]
              cropImg_color_transpose[type=flowunit, flowunit=packed_planar_transpose, device=cpu, deviceid=0, label="<in_image> | <out_image>", queue_size = 16, batch_size = 1]
              cropImg_mean[type=flowunit, flowunit=mean, device=cpu, deviceid=0, label="<in_data> | <out_data>", mean="124, 116, 104", queue_size = 16, batch_size = 8]
              cropImg_normalize[type=flowunit, flowunit=normalize, device=cpu, deviceid=0, label="<in_data> | <out_data>", normalize="0.229, 0.224, 0.225", queue_size = 16, batch_size = 8]
              reid_inference[type=flowunit, flowunit=pedestrian_reid_inference, device=cuda, deviceid=0, label="<input> | <output>", queue_size = 16, batch_size = 8]
              reid_postprocess[type=flowunit, flowunit=reid_postprocess, device=cpu, deviceid=0, label="<embedding> | <bboxes> | <Out_1>", input_width=128, input_height=256, queue_size = 16, batch_size = 8]
              collapse_bbox[type=flowunit, flowunit=collapse_person_bbox, device=cpu, deviceid=0, label="<input> | <output>"]

              pedestrian_yolov3_post: Out_1 ->  expand_bbox_img: In_bbox
              skip_frame:output_frame -> expand_bbox_img: In_img
              expand_bbox_img: Out_img -> has_bbox_condition: In_img
              expand_bbox_img: Out_bbox -> has_bbox_condition: In_bbox
              has_bbox_condition: Out_true -> split_img_bbox: In_true
              has_bbox_condition: Out_false -> collapse_bbox: input
              split_img_bbox: Out_img -> cropImg_resize: in_image
              split_img_bbox: Out_bbox -> reid_postprocess: bboxes
              cropImg_resize: out_image  -> cropImg_color_transpose: in_image
              cropImg_color_transpose: out_image  ->  cropImg_mean: in_data
              cropImg_mean: out_data -> cropImg_normalize: in_data
              cropImg_normalize: out_data -> reid_inference: input
              reid_inference: "output" -> reid_postprocess: embedding
              reid_postprocess: Out_1 -> collapse_bbox: input

              matching[type=flowunit, flowunit=matching, device=cpu, deviceid=0, label="<Input_1> | <Output>", queue_size = 16, batch_size = 1]
              draw_bbox[type=flowunit, flowunit=draw_bbox_mot, device=cpu, deviceid=0, label="<In_1> | <In_2> | <Out_1>", queue_size = 16, batch_size = 5]
              videoencoder[type=flowunit, flowunit=video_encoder, device=cpu, deviceid=0, label="<in_video_frame>", queue_size=16, default_dest_url="rtsp://localhost/test", encoder="mpeg4"]

              collapse_bbox: output -> matching: Input_1
              matching: Output -> draw_bbox: In_1
              skip_frame:output_frame -> draw_bbox: In_2
              draw_bbox: Out_1 -> videoencoder: in_video_frame

            }"""
    format = "graphviz"
  )";

  // run graph
  TestRunGraph(toml_content);
}
}  // namespace modelbox