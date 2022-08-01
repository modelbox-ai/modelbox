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


#ifndef MODELBOX_FLOWUNIT_NORMALIZE_H_
#define MODELBOX_FLOWUNIT_NORMALIZE_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include "modelbox/device/cuda/device_cuda.h"

constexpr const char *FLOWUNIT_NAME = "image_preprocess";
constexpr const char *FLOWUNIT_TYPE = "cuda";
constexpr const char *FLOWUNIT_DESC =
    "A cuda normalize flowunit, the operator is used to normalize for the "
    "image(RGB/BGR).";


class NormalizeFlowUnitV2 : public modelbox::CudaFlowUnit {
 public:
  NormalizeFlowUnitV2();
  ~NormalizeFlowUnitV2() override;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override { return modelbox::STATUS_OK; }

  /* run when processing data */
  modelbox::Status CudaProcess(std::shared_ptr<modelbox::DataContext> data_ctx,
                               cudaStream_t stream) override;

  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  }

  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  }

  modelbox::Status DataGroupPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

  modelbox::Status DataGroupPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override {
    return modelbox::STATUS_OK;
  };

 private:
  std::string output_layout_;
  std::string output_dtype_;
  std::vector<float> mean_;
  std::vector<float> std_;

  std::shared_ptr<modelbox::Buffer> mean_buffer_;
  std::shared_ptr<modelbox::Buffer> std_buffer_;
};

#endif  // MODELBOX_FLOWUNIT_NORMALIZE_H_