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

#include "rknpu2_inference.h"

#include <model_decrypt.h>
#include <modelbox/base/log.h>
#include <modelbox/base/status.h>

#include <algorithm>

#include "modelbox/device/rockchip/rockchip_memory.h"
#include "securec.h"

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

static std::map<std::string, rknn_tensor_type> type_map = {
    {"FLOAT", RKNN_TENSOR_FLOAT32},   {"INT", RKNN_TENSOR_INT32},
    {"FLOAT32", RKNN_TENSOR_FLOAT32}, {"FLOAT16", RKNN_TENSOR_FLOAT16},
    {"INT8", RKNN_TENSOR_INT8},       {"UINT8", RKNN_TENSOR_UINT8},
    {"INT16", RKNN_TENSOR_INT16},     {"UINT16", RKNN_TENSOR_UINT16},
    {"INT32", RKNN_TENSOR_INT32},     {"UINT32", RKNN_TENSOR_UINT32},
    {"INT64", RKNN_TENSOR_INT64}};

static std::map<rknn_tensor_type, size_t> type_size_map = {
    {RKNN_TENSOR_FLOAT32, 4}, {RKNN_TENSOR_FLOAT16, 2}, {RKNN_TENSOR_INT8, 1},
    {RKNN_TENSOR_UINT8, 1},   {RKNN_TENSOR_INT16, 2},   {RKNN_TENSOR_UINT16, 2},
    {RKNN_TENSOR_INT32, 4},   {RKNN_TENSOR_UINT32, 4},  {RKNN_TENSOR_INT64, 8}};

modelbox::Status modelbox::RKNPU2Inference::LoadModel(
    const std::string &model_file,
    const std::shared_ptr<modelbox::Drivers> &drivers_ptr,
    const std::shared_ptr<modelbox::Configuration> &config) {
  ModelDecryption rknpu2_model_decrypt;
  if (modelbox::STATUS_SUCCESS !=
      rknpu2_model_decrypt.Init(model_file, drivers_ptr, config)) {
    MBLOG_ERROR << "init model fail";
    return modelbox::STATUS_FAULT;
  }

  int64_t model_len = 0;
  std::shared_ptr<uint8_t> modelBuf =
      rknpu2_model_decrypt.GetModelSharedBuffer(model_len);
  if (!modelBuf) {
    MBLOG_ERROR << "GetDecryptModelBuffer fail";
    return modelbox::STATUS_FAULT;
  }

  int ret = rknn_init(&ctx_, modelBuf.get(), model_len, 0, nullptr);
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "rknn_init fail:" << ret;
    ctx_ = 0;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status modelbox::RKNPU2Inference::ConvertType(
    const std::string &type, rknn_tensor_type &rk_type) {
  auto tmp_type = type;
  std::transform(tmp_type.begin(), tmp_type.end(), tmp_type.begin(), ::toupper);
  auto iter = type_map.find(tmp_type);
  if (iter == type_map.end()) {
    MBLOG_ERROR << "Not support type: " << type;
    return modelbox::STATUS_FAULT;
  }
  rk_type = iter->second;
  return modelbox::STATUS_OK;
}

modelbox::Status modelbox::RKNPU2Inference::GetModelAttr() {
  inputs_type_.resize(npu2model_input_list_.size());
  inputs_size_.resize(npu2model_input_list_.size());
  // rknn_tensor_attr use new to avoid stack crash
  std::shared_ptr<rknn_tensor_attr> tmp_attr =
      std::make_shared<rknn_tensor_attr>();
  for (size_t i = 0; i < npu2model_input_list_.size(); i++) {
    tmp_attr->index = (unsigned int)i;
    auto ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, tmp_attr.get(),
                          sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      MBLOG_ERROR << "query input attrs error";
      return {modelbox::STATUS_FAULT, "query input attrs error"};
    }

    rknn_tensor_type rk_type;
    auto status = ConvertType(npu2model_type_list_[i], rk_type);
    if (status != modelbox::STATUS_OK) {
      MBLOG_ERROR << "input type convert failed. " << status.WrapErrormsgs();
      return {status, "input type convert failed."};
    }
    inputs_type_[i] = rk_type;
    inputs_size_[i] = tmp_attr->n_elems * type_size_map[rk_type];
  }

  outputs_size_.resize(npu2model_output_list_.size());
  for (size_t i = 0; i < npu2model_output_list_.size(); i++) {
    tmp_attr->index = (unsigned int)i;
    auto ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, tmp_attr.get(),
                          sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      MBLOG_ERROR << "query output attrs error";
      return {modelbox::STATUS_FAULT, "query output attrs error"};
    }

    rknn_tensor_type rk_type;
    auto status = ConvertType(npu2model_type_list_output_[i], rk_type);
    if (status != modelbox::STATUS_OK) {
      MBLOG_ERROR << "output type convert failed. " << status.WrapErrormsgs();
      return {status, "output type convert failed."};
    }
    outputs_size_[i] = tmp_attr->n_elems * type_size_map[rk_type];
  }
  return STATUS_SUCCESS;
}

modelbox::Status modelbox::RKNPU2Inference::Init(
    const std::string &model_file,
    const std::shared_ptr<modelbox::Drivers> &drivers_ptr,
    const std::shared_ptr<modelbox::Configuration> &config,
    const std::shared_ptr<modelbox::InferenceRKNPUParams> &params) {
  batch_size_ = config->GetInt32("batch_size", 1);

  if (LoadModel(model_file, drivers_ptr, config) != STATUS_SUCCESS) {
    return modelbox::STATUS_FAULT;
  }
  // just use input name without check
  npu2model_input_list_ = params->input_name_list_;
  npu2model_type_list_ = params->input_type_list_;
  npu2model_output_list_ = params->output_name_list_;
  npu2model_type_list_output_ = params->output_type_list_;

  rknn_input_output_num rknpu2_io_num;
  auto ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &rknpu2_io_num,
                        sizeof(rknpu2_io_num));
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "query input_output error";
    return {modelbox::STATUS_FAULT, "query input_output error"};
  }

  if (npu2model_input_list_.size() != rknpu2_io_num.n_input ||
      npu2model_output_list_.size() != rknpu2_io_num.n_output) {
    MBLOG_ERROR << "model input output num mismatch: input num in graph is "
                << npu2model_input_list_.size()
                << ", the real model input num is " << rknpu2_io_num.n_input
                << ", output num in graph is " << npu2model_output_list_.size()
                << "the real model output num is " << rknpu2_io_num.n_output;
    return modelbox::STATUS_FAULT;
  }

  return GetModelAttr();
}

modelbox::Status modelbox::RKNPU2Inference::Build_Outputs(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto out_cnt = npu2model_output_list_.size();
  std::vector<rknn_output> rknpu2_outputs;
  rknpu2_outputs.reserve(out_cnt);

  for (size_t i = 0; i < out_cnt; ++i) {
    auto &name = npu2model_output_list_[i];
    auto buffer_list = data_ctx->Output(name);

    std::vector<size_t> shape({outputs_size_[i]});
    buffer_list->Build(shape, false);
    auto rknpu2_buffer = buffer_list->At(0);
    auto *mpp_buf = (MppBuffer)(rknpu2_buffer->MutableData());
    auto *data_buf = (float *)mpp_buffer_get_ptr(mpp_buf);

    // convert outputs to float*
    rknpu2_outputs.push_back({.want_float = true,
                              .is_prealloc = true,
                              .index = (unsigned int)i,
                              .buf = data_buf,
                              .size = (uint32_t)outputs_size_[i]});
    rknpu2_buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);
    rknpu2_buffer->Set("shape", outputs_size_[i]);
  }

  auto ret = rknn_outputs_get(ctx_, out_cnt, rknpu2_outputs.data(), nullptr);
  // reset rknpu2_outputs, avoid buf released
  for (auto ele : rknpu2_outputs) {
    ele.is_prealloc = 1;
    ele.buf = nullptr;
  }
  rknn_outputs_release(ctx_, out_cnt, rknpu2_outputs.data());
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "rknn get output error";
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

size_t modelbox::RKNPU2Inference::CopyFromAlignMemory(
    std::shared_ptr<modelbox::BufferList> &input_buf_list,
    std::shared_ptr<uint8_t> &pdst,
    std::shared_ptr<modelbox::InferenceInputParams> &input_params) {
  int32_t one_size = input_params->in_width_ * input_params->in_height_;
  RgaSURF_FORMAT rga_fmt = RK_FORMAT_UNKNOWN;
  rga_fmt = modelbox::GetRGAFormat(input_params->pix_fmt_);

  size_t input_total_size = batch_size_ * one_size;
  if (rga_fmt == RK_FORMAT_YCbCr_420_SP || rga_fmt == RK_FORMAT_YCrCb_420_SP) {
    input_total_size = input_total_size * 3 / 2;
  } else {
    if (rga_fmt == RK_FORMAT_RGB_888 || rga_fmt == RK_FORMAT_BGR_888) {
      input_total_size = input_total_size * 3;
    }
  }

  if ((batch_size_ == 1) &&
      ((input_params->in_width_ == input_params->in_wstride_ &&
        input_params->in_height_ == input_params->in_hstride_) ||
       (input_params->in_wstride_ == 0 && input_params->in_hstride_ == 0))) {
    auto in_image = input_buf_list->At(0);
    auto *mpp_buf = (MppBuffer)(in_image->ConstData());
    auto *cpu_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_buf);
    pdst.reset(cpu_buf, [](uint8_t *p) {});
    return input_total_size;
  }

  pdst.reset(new u_int8_t[input_total_size],
             [](const uint8_t *p) { delete[] p; });

  uint8_t *pdst_buf = pdst.get();
  for (size_t i = 0; i < batch_size_; i++) {
    auto in_image = input_buf_list->At(i);
    auto *mpp_buf = (MppBuffer)(in_image->ConstData());
    auto *cpu_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_buf);

    if (rga_fmt == RK_FORMAT_YCbCr_420_SP ||
        rga_fmt == RK_FORMAT_YCrCb_420_SP) {
      modelbox::CopyNVMemory(
          cpu_buf, pdst_buf, input_params->in_width_, input_params->in_height_,
          input_params->in_wstride_, input_params->in_hstride_);
      pdst_buf += one_size * 3 / 2;
    } else if (rga_fmt == RK_FORMAT_RGB_888 || rga_fmt == RK_FORMAT_BGR_888) {
      modelbox::CopyRGBMemory(
          cpu_buf, pdst_buf, input_params->in_width_, input_params->in_height_,
          input_params->in_wstride_, input_params->in_hstride_);
      pdst_buf += one_size * 3;
    } else {
      auto rc = memcpy_s(pdst_buf, one_size, cpu_buf, one_size);
      if (rc != EOK) {
        MBLOG_WARN << "RKNPUInference2 copy fail";
      }
      pdst_buf += one_size;
    }
  }

  return input_total_size;
}

size_t modelbox::RKNPU2Inference::GetInputBuffer(
    std::shared_ptr<uint8_t> &input_buf,
    std::shared_ptr<modelbox::BufferList> &input_buf_list) {
  auto in_image = input_buf_list->At(0);
  auto input_params = std::make_shared<InferenceInputParams>();
  input_params->pix_fmt_ = "";

  in_image->Get("width", input_params->in_width_);
  in_image->Get("height", input_params->in_height_);
  in_image->Get("width_stride", input_params->in_wstride_);
  in_image->Get("height_stride", input_params->in_hstride_);
  in_image->Get("pix_fmt", input_params->pix_fmt_);
  if (input_params->pix_fmt_ == "rgb" || input_params->pix_fmt_ == "bgr") {
    input_params->in_wstride_ /= 3;
  } else if (input_params->pix_fmt_.empty()) {
    input_params->in_height_ = 1;
    input_params->in_width_ = in_image->GetBytes();
  }

  if (input_buf_list->GetDevice()->GetType() == "rknpu") {
    return CopyFromAlignMemory(input_buf_list, input_buf, input_params);
  }
  input_buf.reset((uint8_t *)input_buf_list->ConstData(), [](uint8_t *p) {});
  return input_buf_list->GetBytes();
}

modelbox::Status modelbox::RKNPU2Inference::Infer(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  // 构造impl的输入
  if (ctx_ == 0) {
    MBLOG_ERROR << "rk model not load, pass";
    return {STATUS_FAULT, "rk model not load, pass"};
  }

  std::vector<rknn_input> rknpu2_inputs;
  rknpu2_inputs.reserve(npu2model_input_list_.size());
  std::vector<std::shared_ptr<uint8_t>> rknpu2_input_bufs;
  rknpu2_input_bufs.resize(npu2model_input_list_.size());

  for (size_t i = 0; i < npu2model_input_list_.size(); i++) {
    auto inputs = data_ctx->Input(npu2model_input_list_[i]);
    rknn_input one_input;
    size_t realBatch = inputs->Size();
    if (realBatch != batch_size_) {
      auto msg = npu2model_input_list_[i] +
                 " batch mismatch:" + std::to_string(batch_size_) + " " +
                 std::to_string(realBatch);
      MBLOG_ERROR << msg;
      return {STATUS_FAULT, msg};
    }

    size_t ret_size = GetInputBuffer(rknpu2_input_bufs[i], inputs);
    one_input.index = i;
    one_input.buf = rknpu2_input_bufs[i].get();
    one_input.size = ret_size;
    if (one_input.size != inputs_size_[i]) {
      MBLOG_ERROR << "input size mismatch:(yours model) " << one_input.size
                  << " " << inputs_size_[i];
      return modelbox::STATUS_FAULT;
    }
    one_input.pass_through = false;
    one_input.type = (rknn_tensor_type)inputs_type_[i];
    one_input.fmt = RKNN_TENSOR_NHWC;
    rknpu2_inputs.push_back(one_input);
  }

  std::lock_guard<std::mutex> lk(rknpu2_infer_mtx_);
  auto ret = rknn_inputs_set(ctx_, rknpu2_inputs.size(), rknpu2_inputs.data());
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "rknn_inputs_set fail: " << ret;
    return modelbox::STATUS_FAULT;
  }

  ret = rknn_run(ctx_, nullptr);
  if (ret != RKNN_SUCC) {
    MBLOG_ERROR << "run error fail: " << ret;
    return modelbox::STATUS_FAULT;
  }

  return Build_Outputs(data_ctx);
}

modelbox::Status modelbox::RKNPU2Inference::Deinit() {
  std::lock_guard<std::mutex> lk(rknpu2_infer_mtx_);
  if (ctx_ != 0) {
    // 发现，ctrlc退出的时候 rknn_destroy之前需要等一下，
    // 有可能3568比较慢，不然会导致下一次推理异常
    usleep(1000);
    rknn_destroy(ctx_);
    ctx_ = 0;
  }

  return modelbox::STATUS_SUCCESS;
}
