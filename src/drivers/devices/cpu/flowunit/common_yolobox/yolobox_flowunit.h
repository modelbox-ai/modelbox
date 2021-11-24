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
#include "yolo_helper.h"

class BoundingBox {
 public:
  float x_;
  float y_;
  float w_;
  float h_;
  int32_t category_;
  float score_;

  BoundingBox(float x, float y, float w, float h, int32_t category, float score)
      : x_(x), y_(y), w_(w), h_(h), category_(category), score_(score) {}
  ~BoundingBox() {}
};

constexpr const char *FLOWUNIT_NAME = "yolov3_postprocess";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC = "A cpu yolobox flowunit";
constexpr const char *YOLO_TYPE = "yolov3_postprocess";
constexpr const char *INPUT_WIDTH = "input_width";
constexpr const char *INPUT_HEIGHT = "input_height";
constexpr const char *CLASS_NUM = "class_num";

// will auto fill last value, if len(score_list) != class_num
constexpr const char *SCORE_THRESHOLD = "score_threshold";

// will auto fill last value, if len(nms_list) != class_num
constexpr const char *NMS_THRESHOLD = "nms_threshold";
constexpr const char *YOLO_OUTPUT_LAYER_NUM = "yolo_output_layer_num";
constexpr const char *YOLO_OUTPUT_LAYER_WH = "yolo_output_layer_wh";
constexpr const char *ANCHOR_NUM = "anchor_num";
constexpr const char *ANCHOR_BIASES = "anchor_biases";

// will scale result to input as default
constexpr const char *SCALE_TO_INPUT = "scale_to_input";

class YoloboxFlowUnit : public modelbox::FlowUnit {
 public:
  YoloboxFlowUnit();
  virtual ~YoloboxFlowUnit();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);

  modelbox::Status Close() { return modelbox::STATUS_OK; };

  /* run when processing data */
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> data_ctx);

 private:
  modelbox::Status InitYoloParam(YoloParam &param);

  modelbox::Status ReadTensorData(
      std::vector<std::vector<std::shared_ptr<modelbox::Buffer>>> &tensor_data,
      std::shared_ptr<modelbox::DataContext> &data_ctx);

  modelbox::Status SendBoxData(
      std::vector<std::vector<BoundingBox>> &box_data,
      std::shared_ptr<modelbox::DataContext> &data_ctx);

  std::shared_ptr<YoloHelper> yolo_helper_;
  std::vector<std::string> input_name_list_;
  std::vector<std::string> output_name_list_;
};

class YoloboxFlowUnitDesc : public modelbox::FlowUnitDesc {
 public:
  YoloboxFlowUnitDesc() = default;
  virtual ~YoloboxFlowUnitDesc() = default;
};

class YoloboxFlowUnitFactory : public modelbox::FlowUnitFactory {
 public:
  YoloboxFlowUnitFactory() = default;
  virtual ~YoloboxFlowUnitFactory() = default;

  std::shared_ptr<modelbox::FlowUnit> VirtualCreateFlowUnit(
      const std::string &unit_name, const std::string &unit_type,
      const std::string &virtual_type) override;

  const std::string GetFlowUnitFactoryType() override;
  const std::string GetVirtualType() override;

  std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
  FlowUnitProbe() override;
};

#endif  // MODELBOX_FLOWUNIT_YOLOBOXFLOWUNIT_CPU_H_