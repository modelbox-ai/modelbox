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


#include "yolo_helper.h"

#include <modelbox/base/log.h>

constexpr int32_t CLASS_BACKGROUND = -1;

void YoloHelper::GetBoundingBox(
    const float *single_layer_result, size_t layer_index,
    std::function<void(float x, float y, float w, float h, float box_score,
                       int category)> const &save_box_func) {
  auto output_width = param_.layer_wh_[2 * layer_index];
  auto output_height = param_.layer_wh_[2 * layer_index + 1];
  auto step = output_height * output_width;
  auto anchor_size = (5 + param_.class_num_) * output_height * output_width;
  auto anchor_num = param_.anchor_num_[layer_index];
  int category = 0;
  float score = 0;
  for (size_t anchor_index = 0; anchor_index < anchor_num; ++anchor_index) {
    const auto *anchor_data = single_layer_result + anchor_index * anchor_size;
    for (int32_t h = 0; h < output_height; ++h) {
      for (int32_t w = 0; w < output_width; ++w) {
        auto confidence = Sigmoid(anchor_data[4 * step + h * output_width + w]);
        const auto *score_data = anchor_data + 5 * step + h * output_width + w;
        GetCategoryAndScore(score_data, step, param_.class_num_, category,
                            score);
        if (category == CLASS_BACKGROUND) {
          continue;
        }

        GetOneBoundingBox(anchor_data, category, score * confidence,
                          layer_index, step, h, w, anchor_index, save_box_func);
      }
    }
  }
}

void YoloHelper::GetCategoryAndScore(const float *input, int32_t step,
                                     int32_t class_num, int32_t &category,
                                     float &score) {
  if (class_num == 1) {
    score = 1;
    category = 0;
  } else {
    float max_score = -1;
    int32_t max_score_category = CLASS_BACKGROUND;
    for (int32_t c = 0; c < class_num; ++c) {
      if (input[c * step] > max_score) {
        max_score_category = c;
        max_score = input[c * step];
      }
    }

    float sum = 0;
    for (int c = 0; c < class_num; ++c) {
      auto e = static_cast<float>(exp(input[c * step] - max_score));
      sum += e;
    }

    score = static_cast<float>(
        exp(input[max_score_category * step] - max_score) / sum);
    category = max_score_category;
  }
}

void YoloHelper::GetOneBoundingBox(
    const float *anchor_data, int32_t category, float box_score,
    size_t layer_index, int32_t step, int32_t feature_map_h,
    int32_t feature_map_w, size_t anchor_index,
    std::function<void(float x, float y, float w, float h, float box_score,
                       int32_t category)> const &save_box_func) {
  if (box_score < param_.score_threshold_[category]) {
    return;
  }

  auto feature_width = param_.layer_wh_[2 * layer_index];
  auto feature_height = param_.layer_wh_[2 * layer_index + 1];
  float box_x;
  float box_y;
  float box_w;
  float box_h;
  float x_bias;
  float y_bias;

  auto offset = feature_map_h * feature_width + feature_map_w;
  box_x = (feature_map_w + Sigmoid(anchor_data[offset])) / float(feature_width);
  box_y = (feature_map_h + Sigmoid(anchor_data[step + offset])) /
          float(feature_height);
  x_bias =
      param_.anchor_biases_[GetAnchorBiasesOffset(layer_index, anchor_index)];
  y_bias =
      param_
          .anchor_biases_[GetAnchorBiasesOffset(layer_index, anchor_index) + 1];
  box_w = (float)(exp(anchor_data[2 * step + offset]) * x_bias /
                  param_.input_width_);
  box_h = (float)(exp(anchor_data[3 * step + offset]) * y_bias /
                  param_.input_height_);

  box_x = std::max((box_x - box_w / 2.0F), 0.0F);
  box_y = std::max((box_y - box_h / 2.0F), 0.0F);
  box_w = std::min(box_w, 1 - box_x);
  box_h = std::min(box_h, 1 - box_y);
  if (param_.scale_to_input) {
    box_x = box_x * param_.input_width_;
    box_y = box_y * param_.input_height_;
    box_w = box_w * param_.input_width_;
    box_h = box_h * param_.input_height_;
  }

  if (box_w > 0 && box_h > 0 && box_x < param_.input_width_ &&
      box_y < param_.input_height_) {
    save_box_func(box_x, box_y, box_w, box_h, box_score, category);
  }
}

size_t YoloHelper::GetAnchorBiasesOffset(size_t layer_index,
                                         size_t anchor_index) {
  size_t offset = 0;
  for (size_t li = 0; li < layer_index; ++li) {
    offset += param_.anchor_num_[li] * 2;
  }

  offset += anchor_index * 2;
  return offset;
}