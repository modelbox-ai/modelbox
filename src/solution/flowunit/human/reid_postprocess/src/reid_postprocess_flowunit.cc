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


#include "reid_postprocess_flowunit.h"

#include <math.h>
#include <securec.h>

#include <vector>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

using std::map;
using std::vector;
using namespace modelbox;

const int INPUT_SHAPE_SIZE = 1;

ReidPostprocessFlowUnit::ReidPostprocessFlowUnit(){};

ReidPostprocessFlowUnit::~ReidPostprocessFlowUnit(){};

modelbox::Status ReidPostprocessFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  MBLOG_DEBUG << "ReidPostprocessFlowUnit open ";

  width_ = opts->GetUint32("input_width", 128);
  height_ = opts->GetUint32("input_height", 256);
  if (width_ <= 0 || height_ <= 0) {
    auto errMsg = "input width or height is invalid.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_BADCONF, errMsg};
  }

  // buffer for copy data
  for (auto &layer : out_shape_) {
    int buf_size = layer.c * layer.n;
    layer_buffer_.emplace_back(buf_size, 0.0);
    layer_buffer_size_.push_back(buf_size);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status ReidPostprocessFlowUnit::GetInferLayers(
    std::shared_ptr<modelbox::DataContext> data_ctx,
    std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers) {
  size_t batch_size = 0;
  for (size_t i = 0; i < out_shape_name_.size(); ++i) {
    // check batch size, all input batch size is same
    std::shared_ptr<BufferList> layer = data_ctx->Input(out_shape_name_[i]);
    MBLOG_DEBUG << out_shape_name_[i] << ": " << layer->Size();

    auto cur_batch_size = layer->Size();
    if (cur_batch_size <= 0 ||
        (batch_size != 0 && cur_batch_size != batch_size)) {
      auto errMsg =
          "infer layer is invalid. batch_size:" + std::to_string(batch_size) +
          " layer_name:" + out_shape_name_[i] +
          " cur_batch_size:" + std::to_string(cur_batch_size);
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }
    batch_size = cur_batch_size;

    std::vector<size_t> input_shape;
    if (!layer->At(0)->Get("shape", input_shape)) {
      MBLOG_ERROR
          << "reid_postprocess flowunit can not get input 'shape' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key shape"};
    }

    if (!CheckShape(out_shape_[i], input_shape)) {
      auto errMsg = "input layer shape not same.";
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }

    infer_layers.push_back(layer);
  }

  if (infer_layers.empty()) {
    auto errMsg = "infer layer is empty.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status ReidPostprocessFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_DEBUG << "process image ReidPost";

  auto input1_bufs = data_ctx->Input("bboxes");

  std::vector<std::shared_ptr<BBox>> batch_person_bboxes;
  for (size_t i = 0; i < input1_bufs->Size(); ++i) {
    std::shared_ptr<BBox> b = std::make_shared<BBox>();
    auto input_ret =
        memcpy_s(b.get(), sizeof(BBox),
                 (const char *)(input1_bufs->ConstBufferData(i)), sizeof(BBox));
    if (EOK != input_ret) {
      MBLOG_ERROR << "cpu reid_postprocess failed, input_ret " << input_ret;
      return modelbox::STATUS_FAULT;
    }

    batch_person_bboxes.push_back(b);
  }

  std::vector<std::shared_ptr<BufferList>> infer_layers;
  auto status = GetInferLayers(data_ctx, infer_layers);
  if (!status) {
    return status;
  }

  std::vector<size_t> shape(infer_layers[0]->Size(), sizeof(Person));

  auto output_bufs = data_ctx->Output("Out_1");
  auto shape_ret = output_bufs->Build(shape);
  if (!shape_ret) {
    MBLOG_ERROR << "reid_post : get output memory failed.";
    return modelbox::STATUS_NOMEM;
  }

  for (size_t i = 0; i < shape.size(); ++i) {
    auto out_data = (Person *)(*output_bufs)[i]->MutableData();

    auto infer_outputs = infer_layers[0]->At(i)->ConstData();
    auto tmp_buffer_size = infer_layers[0]->At(i)->GetBytes();

    auto emb_ret = memcpy_s(&out_data->emb, tmp_buffer_size, infer_outputs,
                            tmp_buffer_size);
    auto bbox_ret =
        memcpy_s(&out_data->personBox, sizeof(BBox),
                 (const char *)(input1_bufs->ConstBufferData(i)), sizeof(BBox));

    if (EOK != emb_ret || EOK != bbox_ret) {
      MBLOG_ERROR << "cpu reid_postprocess failed, emb_ret " << emb_ret
                  << "bbox_ret" << bbox_ret;
      return modelbox::STATUS_FAULT;
    }
  }

  return modelbox::STATUS_OK;
}

bool ReidPostprocessFlowUnit::CheckShape(ImageShape shape,
                                         const std::vector<size_t> &input) {
  if (input.size() != INPUT_SHAPE_SIZE) {
    MBLOG_ERROR << "input is invalid. input size: " << input.size();
    return false;
  }
  if ((size_t)shape.c != input[0]) {
    MBLOG_ERROR << "input is invalid. shape: " << shape.w << " " << shape.h
                << " " << shape.c << " " << shape.n;
    MBLOG_ERROR << "input is invalid. input: " << input[0] << " " << input[1]
                << " " << input[2];
    return false;
  }
  return true;
}

MODELBOX_FLOWUNIT(ReidPostprocessFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput(
      modelbox::FlowUnitInput("embedding", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("bboxes", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_1", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("input_width", "int", true,
                                                "128", "reid input width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("input_height", "int", true,
                                                "256", "reid input height"));
  desc.SetFlowType(modelbox::NORMAL);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
