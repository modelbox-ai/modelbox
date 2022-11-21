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

#include "mpp_to_cpu_flowunit.h"

#include "modelbox/base/status.h"
#include "modelbox/device/rockchip/rockchip_memory.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

MppToCpuFlowUnit::MppToCpuFlowUnit() = default;
MppToCpuFlowUnit::~MppToCpuFlowUnit() = default;

modelbox::Status MppToCpuFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status MppToCpuFlowUnit::Close() { return modelbox::STATUS_SUCCESS; }

std::shared_ptr<modelbox::Buffer> MppToCpuFlowUnit::ProcessOneImage(
    const std::shared_ptr<modelbox::Buffer> &in_img, std::string &pix_fmt,
    int32_t w, int32_t h, int32_t ws, int32_t hs) {
  RgaSURF_FORMAT rga_fmt = RK_FORMAT_UNKNOWN;
  rga_fmt = modelbox::GetRGAFormat(pix_fmt);
  if (rga_fmt == RK_FORMAT_UNKNOWN) {
    MBLOG_ERROR << "unsupport pix format, pix_fmt: " << pix_fmt;
    return nullptr;
  }

  auto *mpp_buf = (MppBuffer)(in_img->ConstData());
  auto *cpu_buf = (uint8_t *)mpp_buffer_get_ptr(mpp_buf);

  auto device = this->GetBindDevice();
  auto out_img = std::make_shared<modelbox::Buffer>(device);
  out_img->CopyMeta(in_img);

  if ((w == ws && h == hs) || (ws == 0 && hs == 0)) {
    out_img->Build((void *)cpu_buf, in_img->GetBytes(), [](void *p) {});
    out_img->Set("origin_buf", in_img);
    return out_img;
  }

  size_t total_size = 0;
  int32_t div = 1;
  auto ret = modelbox::STATUS_OK;
  if (rga_fmt == RK_FORMAT_YCbCr_420_SP || rga_fmt == RK_FORMAT_YCrCb_420_SP) {
    total_size = w * h * 3 / 2;
    out_img->Build(total_size);
    ret = modelbox::CopyNVMemory(cpu_buf, (uint8_t *)out_img->MutableData(), w,
                                 h, ws, hs);
    div = 2;
  } else {
    total_size = w * h * 3;
    out_img->Build(total_size);
    ret = modelbox::CopyRGBMemory(cpu_buf, (uint8_t *)out_img->MutableData(), w,
                                  h, ws, hs);
  }

  if (ret != modelbox::STATUS_OK) {
    MBLOG_ERROR << "copy image fail, reason: " << ret.Errormsg();
    return nullptr;
  }

  out_img->Set("width", (int32_t)w);
  out_img->Set("height", (int32_t)h);
  if (RK_FORMAT_BGR_888 == rga_fmt || RK_FORMAT_RGB_888 == rga_fmt) {
    out_img->Set("width_stride", (int32_t)(w * 3));
  } else {
    out_img->Set("width_stride", (int32_t)w);
  }
  out_img->Set("height_stride", (int32_t)h);
  int32_t channel = 0;
  in_img->Get("channel", channel);
  out_img->Set("shape", std::vector<size_t>{(size_t)h * 3 / div, (size_t)w,
                                            (size_t)channel});
  return out_img;
}

modelbox::Status MppToCpuFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto input_img_bufs = data_ctx->Input(IN_IMG);
  auto output_bufs = data_ctx->Output(OUT_IMG);

  for (size_t i = 0; i < input_img_bufs->Size(); ++i) {
    std::shared_ptr<modelbox::Buffer> in_img = input_img_bufs->At(i);
    std::string pix_fmt;

    int32_t w = 0;
    int32_t h = 0;
    int32_t ws = 0;
    int32_t hs = 0;
    in_img->Get("pix_fmt", pix_fmt);

    in_img->Get("width", w);
    in_img->Get("height", h);
    in_img->Get("width_stride", ws);
    in_img->Get("height_stride", hs);

    auto out_img = ProcessOneImage(in_img, pix_fmt, w, h, ws, hs);
    if (out_img == nullptr) {
      auto msg = "transfer image to cpu failed, index is " + std::to_string(i);
      MBLOG_ERROR << msg;
      auto buffer = std::make_shared<modelbox::Buffer>();
      buffer->SetError("MppToCpu.Failed", msg);
      output_bufs->PushBack(buffer);
      continue;
    }

    output_bufs->PushBack(out_img);
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(MppToCpuFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({IN_IMG, modelbox::DEVICE_TYPE});
  desc.AddFlowUnitOutput({OUT_IMG, "cpu"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType("cpu");
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion(MODELBOX_VERSION_STR_MACRO);
}