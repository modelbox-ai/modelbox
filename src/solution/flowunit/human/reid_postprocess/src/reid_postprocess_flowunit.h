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


#ifndef MODELBOX_FLOWUNIT_REID_POSTPROCESS_CPU_H_
#define MODELBOX_FLOWUNIT_REID_POSTPROCESS_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "modelbox/buffer.h"
#include "modelbox/flowunit.h"

const int EMBEDDING_LENGTH = 512;

typedef struct tag_BBox {
  float x, y, w, h;
  int category;
  float score;
} BBox;

typedef struct tag_Person {
  float emb[EMBEDDING_LENGTH]{0};
  BBox personBox;
} Person;

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

constexpr const char *FLOWUNIT_NAME = "reid_postprocess";
constexpr const char *FLOWUNIT_DESC = R"(A cpu reid_postprocess flowunit)";

class ReidPostprocessFlowUnit : public modelbox::FlowUnit {
 public:
  ReidPostprocessFlowUnit();
  ~ReidPostprocessFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close() { return modelbox::STATUS_OK; };
  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  bool CheckShape(ImageShape shape, const std::vector<size_t> &input);

  modelbox::Status GetInferLayers(
      std::shared_ptr<modelbox::DataContext> data_ctx,
      std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers);

 private:
  std::vector<std::string> out_shape_name_{"embedding"};

  int width_{128};
  int height_{256};

  std::vector<ImageShape> out_shape_ = {{.w = -1, .h = -1, .c = 512, .n = 1}};

  std::vector<std::vector<float>> layer_buffer_;
  std::vector<int> layer_buffer_size_;
};

#endif  // MODELBOX_FLOWUNIT_REID_POSTPROCESS_CPU_H_