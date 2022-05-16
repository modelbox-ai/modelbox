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

#include "image_rotate.h"

#include "modelbox/flowunit_api_helper.h"

modelbox::Status ImageRotateCpuFlowUnit::RotateOneImage(
    std::shared_ptr<modelbox::Buffer> input_buffer,
    std::shared_ptr<modelbox::Buffer> output_buffer, int32_t rotate_angle,
    int32_t width, int32_t height) {
  cv::Mat input_img(cv::Size(width, height), CV_8UC3,
                    const_cast<void *>(input_buffer->ConstData()));
  auto output_img = std::make_shared<cv::Mat>();
  cv::rotate(input_img, *output_img, rotate_code_[rotate_angle]);

  // build output buffer
  auto ret = output_buffer->BuildFromHost(
      output_img->data, output_img->total() * output_img->elemSize(),
      [output_img](void *ptr) {
        /* Only capture image */
      });
  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "BuildFromHost failed, ret " << ret;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(ImageRotateCpuFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitOutput({"out_image"});

  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "rotate_angle", "int", false, "0", "the image rotate image"));

  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
