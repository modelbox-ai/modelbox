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


#include "color_transpose.h"

#include <unordered_map>

#include "modelbox/flowunit_api_helper.h"
#include "color_transpose_cu.h"

const std::set<std::string> SupportPixFormat = {"bgr", "rgb", "gray"};

NppStatus RGBToGRAY(NppiSize &size, const uint8_t *input_data,
                    uint8_t *output_data, cudaStream_t stream) {
  const int nStepInput = COLOR_CHANNEL_COUNT * size.width;
  const int nStepOutput = GRAY_CHANNEL_COUNT * size.width;

  return nppiRGBToGray_8u_C3C1R(input_data, nStepInput, output_data,
                                nStepOutput, size);
}

NppStatus BGRToGRAY(NppiSize &size, const uint8_t *input_data,
                    uint8_t *output_data, cudaStream_t stream) {
  const int nStepInput = COLOR_CHANNEL_COUNT * size.width;
  const int nStepOutput = GRAY_CHANNEL_COUNT * size.width;

  const Npp32f aCoefs[COLOR_CHANNEL_COUNT] = {0.114f, 0.587f, 0.299f};
  return nppiColorToGray_8u_C3C1R(input_data, nStepInput, output_data,
                                  nStepOutput, size, aCoefs);
}

typedef NppStatus (*pColorTranspose)(NppiSize &, const uint8_t *, uint8_t *,
                                     cudaStream_t);
const std::unordered_map<std::string, pColorTranspose> FunctionTable = {
    {"rgb_to_bgr", RGBToBGR},   {"bgr_to_rgb", BGRToRGB},
    {"rgb_to_gray", RGBToGRAY}, {"bgr_to_gray", BGRToGRAY},
    {"gray_to_rgb", GRAYToRGB}, {"gray_to_bgr", GRAYToBGR},
};

NppStatus ColorTransposeFunction(const std::string &source_color,
                                 const std::string &target_color,
                                 NppiSize &size, const uint8_t *input,
                                 uint8_t *output, cudaStream_t stream) {
  std::string key = source_color + "_" + "to" + "_" + target_color;
  auto iter = FunctionTable.find(key);
  if (iter == FunctionTable.end()) {
    MBLOG_WARN << "can not find transpose function for " << key;
    return NPP_ERROR;
  }

  return iter->second(size, input, output, stream);
}

modelbox::Status ColorTransposeFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  if (!opts->Contain("out_pix_fmt")) {
    MBLOG_ERROR << "config must has out_pix_fmt";
    return modelbox::STATUS_BADCONF;
  }

  out_pix_fmt_ = opts->GetString("out_pix_fmt", "");
  if (SupportPixFormat.find(out_pix_fmt_) == SupportPixFormat.end()) {
    MBLOG_ERROR << "Invalid config out_pix_fmt = " << out_pix_fmt_;
    return modelbox::STATUS_BADCONF;
  }

  return modelbox::STATUS_OK;
}

bool IsColor(const std::string &type) {
  return type == "rgb" || type == "bgr" || type == "ycbcr";
}

std::size_t NumberOfChannels(const std::string &type) {
  return IsColor(type) ? COLOR_CHANNEL_COUNT : GRAY_CHANNEL_COUNT;
}

modelbox::Status GetParm(std::shared_ptr<modelbox::Buffer> buffer,
                       std::vector<size_t> &shape, std::string &input_layout,
                       modelbox::ModelBoxDataType &type, std::string &in_pix_fmt) {
  if (!buffer->Get("shape", shape)) {
    MBLOG_ERROR << "can not get shape from buffer";
    return modelbox::STATUS_FAULT;
  }

  if (shape.size() != GRAY_CHANNEL_COUNT &&
      shape.size() != COLOR_CHANNEL_COUNT) {
    MBLOG_ERROR << "unsupport image shape: " << shape.size();
    return modelbox::STATUS_INVALID;
  }

  if (!buffer->Get("layout", input_layout)) {
    MBLOG_ERROR << "can not get layout from buffer";
    return modelbox::STATUS_INVALID;
  }

  if (input_layout != "hwc") {
    MBLOG_ERROR << "unsupport layout: " << input_layout << " support hwc";
    return modelbox::STATUS_INVALID;
  }

  if (!buffer->Get("type", type)) {
    MBLOG_ERROR << "can not get type from buffer";
    return modelbox::STATUS_INVALID;
  }

  if (type != modelbox::ModelBoxDataType::MODELBOX_UINT8) {
    MBLOG_ERROR << "unsupport type: " << type
                << " support modelbox::ModelBoxDataType::MODELBOX_UINT8";
    return modelbox::STATUS_INVALID;
  }

  if (!buffer->Get("pix_fmt", in_pix_fmt)) {
    MBLOG_ERROR << "can not get pix_fmt from buffer";
    return modelbox::STATUS_INVALID;
  }

  if (SupportPixFormat.find(in_pix_fmt) == SupportPixFormat.end()) {
    MBLOG_ERROR << "Invalid config in_pix_fmt = " << in_pix_fmt;
    return modelbox::STATUS_INVALID;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status GetAndCheckParm(std::shared_ptr<modelbox::BufferList> input,
                               std::vector<size_t> &shape,
                               std::string &input_layout,
                               modelbox::ModelBoxDataType &type,
                               std::string &in_pix_fmt) {
  std::vector<size_t> tmp_shape;
  std::string tmp_input_layout;
  modelbox::ModelBoxDataType tmp_type;
  std::string tmp_in_pix_fmt;

  for (auto &buffer : *input) {
    if (buffer == *input->begin()) {
      if (!GetParm(buffer, shape, input_layout, type, in_pix_fmt)) {
        return modelbox::STATUS_INVALID;
      }
    }

    if (!GetParm(buffer, tmp_shape, tmp_input_layout, tmp_type,
                 tmp_in_pix_fmt)) {
      return modelbox::STATUS_INVALID;
    }

    if (tmp_shape != shape) {
      MBLOG_ERROR << "all image must has same shape.";
      return modelbox::STATUS_INVALID;
    }

    if (tmp_input_layout != input_layout) {
      MBLOG_ERROR << "all image must has same layout.";
      return modelbox::STATUS_INVALID;
    }

    if (tmp_type != type) {
      MBLOG_ERROR << "all image must has same type.";
      return modelbox::STATUS_INVALID;
    }

    if (tmp_in_pix_fmt != in_pix_fmt) {
      MBLOG_ERROR << "all image must has same type.";
      return modelbox::STATUS_INVALID;
    }
  }

  return modelbox::STATUS_OK;
}

/* run when processing data */
modelbox::Status ColorTransposeFlowUnit::CudaProcess(
    std::shared_ptr<modelbox::DataContext> data_ctx, cudaStream_t stream) {
  auto input = data_ctx->Input("in_image");
  auto output = data_ctx->Output("out_image");

  std::vector<size_t> shape;
  std::string input_layout;
  modelbox::ModelBoxDataType type = modelbox::MODELBOX_TYPE_INVALID;
  std::string in_pix_fmt;

  auto status = GetAndCheckParm(input, shape, input_layout, type, in_pix_fmt);
  if (!status) {
    return status;
  }

  if (in_pix_fmt == out_pix_fmt_) {
    MBLOG_INFO << "out pix_fmt is same with in pix_fmt.";
    for (unsigned int i = 0; i < input->Size(); ++i) {
      output->PushBack(input->At(i));
    }

    return modelbox::STATUS_OK;
  }

  size_t H = shape[0], W = shape[1];
  size_t output_C = NumberOfChannels(out_pix_fmt_);
  std::vector<size_t> shapes(input->Size(),
                             H * W * output_C * GetDataTypeSize(type));
  output->Build(shapes);

  auto cuda_ret = cudaStreamSynchronize(stream);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "sync stream  " << stream << " failed, err " << cuda_ret;
    return modelbox::STATUS_FAULT;
  }

  for (unsigned int i = 0; i < input->Size(); ++i) {
    NppiSize size;
    size.height = H;
    size.width = W;

    const uint8_t *input_data = (const uint8_t *)(input->At(i)->ConstData());
    uint8_t *output_data = (uint8_t *)(output->At(i)->ConstData());

    auto npp_status = ColorTransposeFunction(in_pix_fmt, out_pix_fmt_, size,
                                             input_data, output_data, stream);
    if (NPP_SUCCESS != npp_status) {
      MBLOG_WARN << "ColorTranspose npp return failed, status: " << npp_status;
      status = modelbox::STATUS_FAULT;
      break;
    }

    output->At(i)->CopyMeta(input->At(i));
    output->At(i)->Set("pix_fmt", out_pix_fmt_);
    output->At(i)->Set("channel", output_C);
    output->At(i)->Set("shape", std::vector<size_t>({H, W, output_C}));
  }

  return status;
}

MODELBOX_FLOWUNIT(ColorTransposeFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("in_image", FLOWUNIT_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("out_image", FLOWUNIT_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);

  std::map<std::string, std::string> pix_fmt_list;

  for (auto &item : SupportPixFormat) {
    pix_fmt_list[item] = item;
  }
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "out_pix_fmt", "list", true, "",
      "the colour transpose output pixel format", pix_fmt_list));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
