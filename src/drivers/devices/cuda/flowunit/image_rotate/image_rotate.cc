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

#include "image_rotate_cu.h"
#include "modelbox/flowunit_api_helper.h"

modelbox::Status ImageRotateGpuFlowUnit::RotateOneImage(
    std::shared_ptr<modelbox::Buffer> input_buffer,
    std::shared_ptr<modelbox::Buffer> output_buffer, int32_t rotate_angle,
    int32_t width, int32_t height) {
  auto cuda_ret = cudaSetDevice(dev_id_);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Set cuda device " << dev_id_ << " failed, cuda ret "
                << cuda_ret;
    return modelbox::STATUS_FAULT;
  }

  std::shared_ptr<modelbox::CudaStream> stream;
  if (GetStream(input_buffer, output_buffer, stream) != modelbox::STATUS_OK) {
    return modelbox::STATUS_FAULT;
  }

  output_buffer->Build(input_buffer->GetBytes());
  auto output_data = static_cast<u_char *>(output_buffer->MutableData());

  auto ret =
      ClockWiseRotateGPU((u_char *)input_buffer->ConstData(), output_data,
                         height, width, rotate_angle, stream->Get());
  if (ret != 0) {
    MBLOG_ERROR << "gpu rotate image failed.";
    return modelbox::STATUS_FAULT;
  }

  stream->Bind({input_buffer->GetDeviceMemory()});
  return modelbox::STATUS_OK;
}

modelbox::Status ImageRotateGpuFlowUnit::GetStream(
    std::shared_ptr<modelbox::Buffer> input_buffer,
    std::shared_ptr<modelbox::Buffer> output_buffer,
    std::shared_ptr<modelbox::CudaStream> &stream) {
  auto input_cuda_mem = std::dynamic_pointer_cast<modelbox::CudaMemory>(
      input_buffer->GetDeviceMemory());
  stream = input_cuda_mem->GetBindStream();
  // bind same stream
  auto output_cuda_mem = std::dynamic_pointer_cast<modelbox::CudaMemory>(
      output_buffer->GetDeviceMemory());
  auto status = output_cuda_mem->BindStream(stream);
  if (status != modelbox::STATUS_OK) {
    auto err_msg = "bind stream failed, " + status.WrapErrormsgs();
    MBLOG_WARN << err_msg;
    return status;
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(ImageRotateGpuFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_origin_image"});
  desc.AddFlowUnitOutput({"out_rotate_image"});

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
