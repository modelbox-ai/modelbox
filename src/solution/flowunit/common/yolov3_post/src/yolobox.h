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


#ifndef MODELBOX_FLOWUNIT_YOLOBOXFLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_YOLOBOXFLOWUNIT_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

typedef struct tag_BBox {
  float x, y, w, h;
  int category;
  float score;
} BBox;

typedef struct tag_OBJECT_DETECT_MODEL_PARAM {
  float *biases;
  int anchor_num;
  int classes;
  float nms_thresh;
  float score_thresh;
} OBJECT_DETECT_MODEL_PARAM;

typedef struct tag_ImageShape {
  int w;
  int h;
  int c;
  int n;
} ImageShape;

constexpr const char *FLOWUNIT_NAME = "yolov3_post";
constexpr const char *FLOWUNIT_DESC = R"(A cpu yolov3_post flowunit)";

class YoloboxFlowUnit : public modelbox::FlowUnit {
 public:
  YoloboxFlowUnit();
  ~YoloboxFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close() { return modelbox::STATUS_OK; };

  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  bool CheckShape(ImageShape shape, std::vector<size_t> input);

  modelbox::Status GetInferLayers(
      std::shared_ptr<modelbox::DataContext> data_ctx,
      std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers);

  std::vector<BBox> ComputerBox(
      std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers,
      int batch_idx);

  modelbox::Status SendBox(std::shared_ptr<modelbox::DataContext> data_ctx,
                         const std::vector<std::vector<BBox>> &out_bboxes);

 private:
  std::vector<std::string> out_shape_name_{"layer82-conv", "layer94-conv",
                                           "layer106-conv"};
  std::vector<ImageShape> out_shape_ = {{.w = 19, .h = 19, .c = 255, .n = 1},
                                        {.w = 38, .h = 38, .c = 255, .n = 1},
                                        {.w = 76, .h = 76, .c = 255, .n = 1}};
  // anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90, 156,198,
  // 373,326
  float biases_[18] = {3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875,
                       1.875, 3.8125, 3.875, 2.8125, 3.6875,   7.4375,
                       1.25,  1.625,  2.0,   3.75,   4.125,    2.875};
  OBJECT_DETECT_MODEL_PARAM param_ = {.biases = biases_,
                                      .anchor_num = 3,
                                      .classes = 80,
                                      .nms_thresh = 0.45,
                                      .score_thresh = 0.6};
  int image_width_{1920};
  int image_height_{1080};
  int width_{608};
  int height_{608};
  std::vector<std::vector<float>> layer_buffer_;
  std::vector<int> layer_buffer_size_;
};
#endif  // MODELBOX_FLOWUNIT_YOLOBOXFLOWUNIT_CPU_H_