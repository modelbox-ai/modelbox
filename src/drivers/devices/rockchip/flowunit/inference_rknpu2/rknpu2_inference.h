/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#ifndef MODELBOX_RKNPU2_INFERENCE_H_
#define MODELBOX_RKNPU2_INFERENCE_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>
#include <modelbox/data_context.h>
#include <modelbox/device/rockchip/device_rockchip.h>
#include <modelbox/device/rockchip/rockchip_api.h>

#include <string>
#include <vector>

#include "rknn_api.h"

namespace modelbox {

class RKNPU2Inference {
 public:
  Status Deinit();
  Status Init(const std::string &model_file,
              const std::shared_ptr<modelbox::Drivers> &drivers_ptr,
              const std::shared_ptr<modelbox::Configuration> &config,
              const std::shared_ptr<modelbox::InferenceRKNPUParams> &params);

  Status Infer(std::shared_ptr<modelbox::DataContext> &data_ctx);

 private:
  size_t GetInputBuffer(std::shared_ptr<uint8_t> &input_buf,
                        std::shared_ptr<modelbox::BufferList> &input_buf_list);
  size_t CopyFromAlignMemory(
      std::shared_ptr<modelbox::BufferList> &input_buf_list,
      std::shared_ptr<uint8_t> &pdst,
      std::shared_ptr<modelbox::InferenceInputParams> &input_params);
  Status Build_Outputs(std::shared_ptr<modelbox::DataContext> &data_ctx);
  Status GetModelAttr();
  Status LoadModel(const std::string &model_file,
                   const std::shared_ptr<modelbox::Drivers> &drivers_ptr,
                   const std::shared_ptr<modelbox::Configuration> &config);
  Status ConvertType(const std::string &type, rknn_tensor_type &rk_type);

  size_t batch_size_{1};
  std::vector<size_t> outputs_size_;
  std::vector<size_t> inputs_size_;
  std::vector<int> inputs_type_;
  std::vector<std::string> npu2model_input_list_;
  std::vector<std::string> npu2model_type_list_;
  std::vector<std::string> npu2model_output_list_;
  std::vector<std::string> npu2model_type_list_output_;
  rknn_context ctx_{0};
  std::mutex rknpu2_infer_mtx_;
};
}  // namespace modelbox
#endif