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

#ifndef MODELBOX_FLOWUNIT_IMAGE_ROTATE_BASE_H_
#define MODELBOX_FLOWUNIT_IMAGE_ROTATE_BASE_H_

#include <modelbox/base/device.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <set>

class ImageRotateFlowUnitBase : public modelbox::FlowUnit {
 public:
  ImageRotateFlowUnitBase();
  virtual ~ImageRotateFlowUnitBase();

  modelbox::Status Open(const std::shared_ptr<modelbox::Configuration> &opts);
  modelbox::Status Close();
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ct);

  virtual modelbox::Status RotateOneImage(
      std::shared_ptr<modelbox::Buffer> input_buffer,
      std::shared_ptr<modelbox::Buffer> output_buffer, int32_t rotate_angle,
      int32_t width, int32_t height) = 0;

 private:
  modelbox::Status CheckImageType(
      std::shared_ptr<modelbox::Buffer> input_buffer);
  modelbox::Status CheckRotateAngle(const int32_t &rotate_angle);

  std::set<int32_t> rotate_value_{90, 180, 270};
  bool has_rotate_angle_{false};
  int32_t rotate_angle_{0};
};

#endif  // MODELBOX_FLOWUNIT_IMAGE_ROTATE_BASE_H_