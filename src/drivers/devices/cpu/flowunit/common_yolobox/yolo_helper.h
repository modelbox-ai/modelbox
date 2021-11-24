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


#ifndef MODELBOX_FLOWUNIT_YOLO_HELPER_H
#define MODELBOX_FLOWUNIT_YOLO_HELPER_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

class YoloParam {
 public:
  int32_t input_width_;
  int32_t input_height_;
  int32_t class_num_;
  std::vector<float> score_threshold_;
  std::vector<float> nms_threshold_;
  int32_t layer_num_;
  std::vector<int32_t> layer_wh_;
  std::vector<uint64_t> anchor_num_;
  std::vector<float> anchor_biases_;
  bool scale_to_input;
};

class YoloHelper {
 public:
  YoloHelper(const YoloParam &param) : param_{param} {}

  virtual ~YoloHelper() = default;

  void GetBoundingBox(
      const float *single_layer_result, size_t layer_index,
      std::function<void(float x, float y, float w, float h, float box_score,
                         int category)> const &save_box_func);

  template <class T>
  void Sort(std::vector<T> &box_list,
            std::function<bool(const T &box1, const T &box2)> const &compare);

  template <class T>
  void NMS(
      std::vector<T> &src_box_list, std::vector<T> &dst_box_list,
      std::function<bool(const T &box1, const T &box2,
                         std::vector<float> &nms_threshold)> const &overlap);

 private:
  inline float Sigmoid(float x) {
    return static_cast<float>(1. / (1. + exp(-x)));
  }

  void GetCategoryAndScore(const float *input, int32_t step, int32_t class_num,
                           int32_t &category, float &score);

  void GetOneBoundingBox(
      const float *anchor_data, int32_t category, float box_score,
      size_t layer_index, int32_t step, int32_t feature_map_h,
      int32_t feature_map_w, size_t anchor_index,
      std::function<void(float x, float y, float w, float h, float box_score,
                         int32_t category)> const &save_box_func);

  size_t GetAnchorBiasesOffset(size_t layer_index, size_t anchor_index);

  YoloParam param_;
};

template <class T>
void YoloHelper::Sort(
    std::vector<T> &box_list,
    std::function<bool(const T &box1, const T &box2)> const &compare) {
  std::sort(box_list.begin(), box_list.end(), compare);
}

template <class T>
void YoloHelper::NMS(
    std::vector<T> &src_box_list, std::vector<T> &dst_box_list,
    std::function<bool(const T &box1, const T &box2,
                       std::vector<float> &nms_threshold)> const &overlap) {
  auto size = src_box_list.size();
  std::unordered_set<size_t> set;
  for (size_t i = 0; i < size; ++i) {
    if (set.find(i) != set.end()) {
      // Has been tested
      continue;
    }

    dst_box_list.push_back(src_box_list[i]);
    // Find box that overlap >= threshold
    for (size_t j = i + 1; j < size; ++j) {
      if (set.find(j) != set.end()) {
        continue;
      }

      if (overlap(src_box_list[i], src_box_list[j], param_.nms_threshold_)) {
        set.insert(j);  // Will not access this box next time
      }
    }
  }
}

#endif  // MODELBOX_FLOWUNIT_YOLO_HELPER_H