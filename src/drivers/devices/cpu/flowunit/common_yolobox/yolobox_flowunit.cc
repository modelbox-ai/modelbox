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


#include "yolobox_flowunit.h"

#include <math.h>
#include <securec.h>

#include <cmath>
#include <vector>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "virtualdriver_yolobox.h"

using namespace modelbox;

YoloboxFlowUnit::YoloboxFlowUnit() = default;

YoloboxFlowUnit::~YoloboxFlowUnit() = default;

modelbox::Status YoloboxFlowUnit::InitYoloParam(YoloParam &param) {
  auto config =
      std::dynamic_pointer_cast<YoloBoxVirtualFlowUnitDesc>(GetFlowUnitDesc())
          ->GetConfiguration();
  param.input_width_ = config->GetInt32(INPUT_WIDTH);
  param.input_height_ = config->GetInt32(INPUT_HEIGHT);
  param.class_num_ = config->GetInt32(CLASS_NUM);
  param.score_threshold_ = config->GetFloats(SCORE_THRESHOLD);
  param.nms_threshold_ = config->GetFloats(NMS_THRESHOLD);
  param.layer_num_ = config->GetInt32(YOLO_OUTPUT_LAYER_NUM);
  param.layer_wh_ = config->GetInt32s(YOLO_OUTPUT_LAYER_WH);
  param.anchor_num_ = config->GetUint64s(ANCHOR_NUM);
  param.anchor_biases_ = config->GetFloats(ANCHOR_BIASES);
  param.scale_to_input = config->GetBool(SCALE_TO_INPUT, true);

  if (param.score_threshold_.empty()) {
    MBLOG_ERROR << SCORE_THRESHOLD << " should not empty";
    return modelbox::STATUS_BADCONF;
  }

  if (param.nms_threshold_.empty()) {
    MBLOG_ERROR << NMS_THRESHOLD << " should not empty";
    return modelbox::STATUS_BADCONF;
  }

  if (param.layer_wh_.size() != (size_t)(param.layer_num_ * 2)) {
    MBLOG_ERROR << YOLO_OUTPUT_LAYER_WH << " size != " << YOLO_OUTPUT_LAYER_NUM
                << " * 2";
    return modelbox::STATUS_BADCONF;
  }

  if (param.anchor_num_.size() != (size_t)param.layer_num_) {
    MBLOG_ERROR << ANCHOR_NUM << " size != " << YOLO_OUTPUT_LAYER_NUM;
    return modelbox::STATUS_BADCONF;
  }

  auto total_anchor = std::accumulate(param.anchor_num_.begin(),
                                      param.anchor_num_.end(), (size_t)0);
  if ((total_anchor * 2) != param.anchor_biases_.size()) {
    MBLOG_ERROR << ANCHOR_BIASES << " size != total anchor number * 2";
    return modelbox::STATUS_BADCONF;
  }

  // Auto fill last value to meet class num, so developer does not need to write
  // same value for all class
  while ((size_t)param.class_num_ > param.score_threshold_.size()) {
    param.score_threshold_.push_back(param.score_threshold_.back());
  }

  while ((size_t)param.class_num_ > param.nms_threshold_.size()) {
    param.nms_threshold_.push_back(param.nms_threshold_.back());
  }

  return STATUS_OK;
}

modelbox::Status YoloboxFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  YoloParam param;
  auto ret = InitYoloParam(param);
  if (!ret) {
    return ret;
  }

  yolo_helper_ = std::make_shared<YoloHelper>(param);
  auto desc = GetFlowUnitDesc();
  auto input_list = desc->GetFlowUnitInput();
  for (auto &input : input_list) {
    input_name_list_.push_back(input.GetPortName());
  }

  if (input_name_list_.empty()) {
    MBLOG_ERROR << "Input is empty";
    return STATUS_BADCONF;
  }

  auto output_list = desc->GetFlowUnitOutput();
  for (auto &output : output_list) {
    output_name_list_.push_back(output.GetPortName());
  }

  if (output_name_list_.size() != 1) {
    MBLOG_ERROR << "Should only has one output port";
    return STATUS_BADCONF;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status YoloboxFlowUnit::ReadTensorData(
    std::vector<std::vector<std::shared_ptr<modelbox::Buffer>>> &tensor_data,
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  tensor_data.clear();
  size_t batch_size = 0;
  for (size_t tensor_index = 0; tensor_index < input_name_list_.size();
       ++tensor_index) {
    auto tensor_name = input_name_list_[tensor_index];
    auto input_buffers = data_ctx->Input(tensor_name);
    if (batch_size == 0) {
      batch_size = input_buffers->Size();
      tensor_data.resize(batch_size);
    } else if (input_buffers->Size() != batch_size) {
      MBLOG_ERROR << "buffers [" << tensor_name << "] size ["
                  << input_buffers->Size() << "] is not same with other["
                  << batch_size << "]";
      return modelbox::STATUS_FAULT;
    }

    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      tensor_data[batch_index].push_back(input_buffers->At(batch_index));
    }
  }

  return modelbox::STATUS_OK;
}

bool Comp(const BoundingBox &box1, const BoundingBox &box2) {
  return box1.score_ > box2.score_;
}

bool Overlap(const BoundingBox &box1, const BoundingBox &box2,
             std::vector<float> &nms_threshold) {
  if (box1.category_ != box2.category_) {
    return false;
  }

  float threshold = nms_threshold[box1.category_];
  float left = std::max(box1.x_, box2.x_);
  float right = std::min(box1.x_ + box1.w_, box2.x_ + box2.w_);
  float top = std::max(box1.y_, box2.y_);
  float down = std::min(box1.y_ + box1.h_, box2.y_ + box2.h_);
  if (left >= right or top >= down) {
    return 0.0F;
  }

  float inter_area = (right - left) * (down - top);
  float union_area = (box1.w_ * box1.h_ + box2.w_ * box2.h_ - inter_area);
  return inter_area >= threshold * union_area;
}

modelbox::Status YoloboxFlowUnit::SendBoxData(
    std::vector<std::vector<BoundingBox>> &box_data,
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  std::vector<size_t> shape;
  for (auto &boxes : box_data) {
    shape.push_back(boxes.size() * sizeof(BoundingBox));
  }

  auto output_buffers = data_ctx->Output(output_name_list_[0]);
  output_buffers->Build(shape);
  for (size_t batch_index = 0; batch_index < box_data.size(); ++batch_index) {
    auto &box_data_for_single_batch = box_data[batch_index];
    auto *box_buffer_ptr =
        (BoundingBox *)(output_buffers->At(batch_index)->MutableData());
    if (box_buffer_ptr == nullptr) {
      continue;
    }

    for (size_t box_index = 0; box_index < box_data_for_single_batch.size();
         ++box_index) {
      box_buffer_ptr[box_index] = box_data_for_single_batch[box_index];
    }
  }

  return modelbox::STATUS_OK;
}

modelbox::Status YoloboxFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  std::vector<std::vector<std::shared_ptr<modelbox::Buffer>>> tensor_data;
  auto ret = ReadTensorData(tensor_data, data_ctx);
  if (!ret) {
    return ret;
  }

  std::vector<std::vector<BoundingBox>> detected_boxes_mul_batch;
  for (size_t batch_index = 0; batch_index < tensor_data.size();
       ++batch_index) {
    std::vector<BoundingBox> detected_boxes_single_batch;
    auto &tensors_in_one_batch = tensor_data[batch_index];
    for (size_t tensor_index = 0; tensor_index < tensors_in_one_batch.size();
         ++tensor_index) {
      auto &tensor = tensors_in_one_batch[tensor_index];
      yolo_helper_->GetBoundingBox(
          (const float *)tensor->ConstData(), tensor_index,
          [&detected_boxes_single_batch](float x, float y, float w, float h,
                                         float box_score, int category) {
            detected_boxes_single_batch.emplace_back(x, y, w, h, category,
                                                     box_score);
          });
    }

    yolo_helper_->Sort<BoundingBox>(detected_boxes_single_batch, Comp);
    std::vector<BoundingBox> final_boxes;
    yolo_helper_->NMS<BoundingBox>(detected_boxes_single_batch, final_boxes,
                                   Overlap);
    detected_boxes_mul_batch.push_back(final_boxes);
  }

  ret = SendBoxData(detected_boxes_mul_batch, data_ctx);
  if (!ret) {
    return ret;
  }

  return modelbox::STATUS_OK;
}

std::string YoloboxFlowUnitFactory::GetFlowUnitFactoryType() {
  return DEVICE_TYPE;
}

std::string YoloboxFlowUnitFactory::GetVirtualType() {
  return YOLO_TYPE;
}

std::shared_ptr<modelbox::FlowUnit> YoloboxFlowUnitFactory::VirtualCreateFlowUnit(
    const std::string &unit_name, const std::string &unit_type,
    const std::string &virtual_type) {
  return std::make_shared<YoloboxFlowUnit>();
}

std::map<std::string, std::shared_ptr<modelbox::FlowUnitDesc>>
YoloboxFlowUnitFactory::FlowUnitProbe() {
  return {};
}
