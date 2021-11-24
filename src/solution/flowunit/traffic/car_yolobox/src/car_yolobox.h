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

 
#ifndef MODELBOX_FLOWUNIT_CARYOLOBOXFLOWUNIT_CPU_H_
#define MODELBOX_FLOWUNIT_CARYOLOBOXFLOWUNIT_CPU_H_
 
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
 
constexpr const char *FLOWUNIT_NAME = "car_yolobox";
constexpr const char *FLOWUNIT_DESC = R"(A cpu yolobox flowunit)";
 
class CarYoloboxFlowUnit : public modelbox::FlowUnit {
 public:
  CarYoloboxFlowUnit();
  ~CarYoloboxFlowUnit();
 
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
  std::vector<std::string> out_shape_name_{"layer15-conv", "layer22-conv"};
  std::vector<ImageShape> out_shape_ = {{.w = 25, .h = 15, .c = 24, .n = 1},
                                        {.w = 50, .h = 30, .c = 24, .n = 1}};
  float biases_[16] = {3.125, 2.25, 5.41, 1.72, 5.16,  4.125, 8.75,   7.875,
                       0.625, 0.5,  1.25, 1.0,  1.875, 1.5,   4.1875, 3.5};
  OBJECT_DETECT_MODEL_PARAM param_ = {.biases = biases_,
                                      .anchor_num = 4,
                                      .classes = 1,
                                      .nms_thresh = 0.45,
                                      .score_thresh = 0.6};
  int image_width_{1920};
  int image_height_{1080};
  int width_{800};
  int height_{480};
  std::vector<std::vector<float>> layer_buffer_;
  std::vector<int> layer_buffer_size_;
};
 
#endif  // MODELBOX_FLOWUNIT_CARYOLOBOXFLOWUNIT_CPU_H_