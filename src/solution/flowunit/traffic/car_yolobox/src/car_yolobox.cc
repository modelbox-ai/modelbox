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


#include "car_yolobox.h"

#include <math.h>
#include <securec.h>

#include <vector>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

using std::map;
using std::vector;
using namespace modelbox;

const int INPUT_SHAPE_SIZE = 3;

static void computeYoloBBox(const OBJECT_DETECT_MODEL_PARAM &model_param,
                            float *output_data, int out_index,
                            vector<BBox> &bboxes, int img_width, int img_height,
                            int input_w, int input_h, int resize_w,
                            int resize_h, int output_w, int output_h,
                            int output_c);

static void ApplyNMSV2(vector<BBox> &bboxes, vector<BBox> &outBBoxes,
                       float nms);

CarYoloboxFlowUnit::CarYoloboxFlowUnit(){};

CarYoloboxFlowUnit::~CarYoloboxFlowUnit(){};

modelbox::Status CarYoloboxFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  MBLOG_DEBUG << "yolobox open ";

  image_width_ = opts->GetUint32("image_width", 1920);
  image_height_ = opts->GetUint32("image_height", 1080);
  if (image_width_ <= 0 || image_height_ <= 0) {
    auto errMsg = "image width or height is invalid.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_BADCONF, errMsg};
  }

  width_ = opts->GetUint32("input_width", 800);
  height_ = opts->GetUint32("input_height", 480);
  if (width_ <= 0 || height_ <= 0) {
    auto errMsg = "input width or height is invalid.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_BADCONF, errMsg};
  }

  // buffer for copy data
  for (auto &layer : out_shape_) {
    int buf_size = layer.w * layer.h * layer.c * layer.n;
    layer_buffer_.emplace_back(buf_size, 0.0);
    layer_buffer_size_.push_back(buf_size);
  }

  return modelbox::STATUS_OK;
}

modelbox::Status CarYoloboxFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_DEBUG << "process image yolobox";

  // checkout input batch_size and shape
  std::vector<std::shared_ptr<BufferList>> infer_layers;
  auto status = GetInferLayers(data_ctx, infer_layers);
  if (!status) {
    return status;
  }

  // batch size
  vector<vector<BBox>> out_bboxes;
  for (size_t batch_idx = 0; batch_idx < infer_layers[0]->Size(); ++batch_idx) {
    auto one_batch_bboxes = ComputerBox(infer_layers, batch_idx);
    out_bboxes.push_back(one_batch_bboxes);
  }

  status = SendBox(data_ctx, out_bboxes);
  if (!status) {
    return status;
  }

  return modelbox::STATUS_OK;
}

bool CarYoloboxFlowUnit::CheckShape(ImageShape shape,
                                    std::vector<size_t> input) {
  if (input.size() != INPUT_SHAPE_SIZE) {
    MBLOG_ERROR << "input is invalid. input size: " << input.size();
    return false;
  }
  if ((size_t)shape.c != input[0] || (size_t)shape.h != input[1] ||
      (size_t)shape.w != input[2]) {
    MBLOG_ERROR << "input is invalid. shape: " << shape.w << " " << shape.h
                << " " << shape.c << " " << shape.n;
    MBLOG_ERROR << "input is invalid. input: " << input[0] << " " << input[1]
                << " " << input[2];
    return false;
  }
  return true;
}

modelbox::Status CarYoloboxFlowUnit::GetInferLayers(
    std::shared_ptr<modelbox::DataContext> data_ctx,
    std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers) {
  size_t batch_size = 0;
  for (size_t i = 0; i < out_shape_name_.size(); ++i) {
    // check batch size, all input batch size is same
    std::shared_ptr<BufferList> layer = data_ctx->Input(out_shape_name_[i]);
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
    layer->At(0)->Get("shape", input_shape);
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

vector<BBox> CarYoloboxFlowUnit::ComputerBox(
    std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers,
    int batch_idx) {
  // one batch
  vector<BBox> one_batch_bboxes;
  vector<BBox> all_layer_bboxes;
  for (size_t i = 0; i < out_shape_.size(); ++i) {
    vector<BBox> layer_bboxes;
    auto infer_outputs = infer_layers[i]->At(batch_idx)->ConstData();
    if (infer_outputs == nullptr) {
      MBLOG_WARN << "infer buffer is null";
      continue;
    }

    auto tmp_buffer_size = infer_layers[i]->At(batch_idx)->GetBytes();
    std::vector<float> tmp(tmp_buffer_size / sizeof(float));
    if (memcpy_s(tmp.data(), tmp_buffer_size, infer_outputs, tmp_buffer_size) !=
        0) {
      MBLOG_WARN << "memory copy failed, size " << tmp_buffer_size;
      continue;
    }

    computeYoloBBox(param_, tmp.data(), i, layer_bboxes, image_width_,
                    image_height_, width_, height_, width_, height_,
                    out_shape_[i].w, out_shape_[i].h, out_shape_[i].c);
    all_layer_bboxes.insert(all_layer_bboxes.end(), layer_bboxes.begin(),
                            layer_bboxes.end());
  }

  // nms
  std::sort(all_layer_bboxes.begin(), all_layer_bboxes.end(),
            [](const BBox &bbox1, const BBox &bbox2) {
              return bbox1.score > bbox2.score;
            });
  ApplyNMSV2(all_layer_bboxes, one_batch_bboxes, param_.nms_thresh);

  // log
  for (auto box : one_batch_bboxes) {
    MBLOG_DEBUG << "batch idx:" << batch_idx << " [" << box.x << "," << box.y
                << "," << box.w << "," << box.h << "] " << box.score << " "
                << box.category;
  }

  return one_batch_bboxes;
}

modelbox::Status CarYoloboxFlowUnit::SendBox(
    std::shared_ptr<modelbox::DataContext> data_ctx,
    const std::vector<std::vector<BBox>> &out_bboxes) {
  // out boxes output shape
  std::vector<size_t> shape;
  for (auto &boxes : out_bboxes) {
    shape.emplace_back(boxes.size() * sizeof(BBox));
  }
  auto output_bufs = data_ctx->Output("Out_1");
  output_bufs->Build(shape);

  // output
  for (size_t i = 0; i < shape.size(); ++i) {
    MBLOG_DEBUG << "batch:" << i << ", size:" << out_bboxes[i].size();

    auto out_data = (BBox *)(*output_bufs)[i]->MutableData();
    if (out_data == nullptr) {
      MBLOG_WARN << "output buffer is invalid, index is " << i;
      continue;
    }

    for (auto &boxes : out_bboxes[i]) {
      *out_data = boxes;
      out_data++;

      MBLOG_DEBUG << "[" << boxes.x << " " << boxes.y << " " << boxes.w << " "
                  << boxes.h << "]"
                  << " score:" << boxes.score
                  << ", category:" << boxes.category;
    }
  }
  return modelbox::STATUS_OK;
}

static float Sigmoid(float x);

static float overlap(BBox bbox1, BBox bbox2);

static void ClassIndexAndScore(float *input, int index, int step, int classes,
                               int &classIdx, float &classScore);

void computeYoloBBox(const OBJECT_DETECT_MODEL_PARAM &model_param,
                     float *output_data, int out_index, vector<BBox> &bboxes,
                     int img_width, int img_height, int input_w, int input_h,
                     int resize_w, int resize_h, int output_w, int output_h,
                     int output_c) {
  if ((0 == resize_h) || (0 == resize_w)) {
    return;
  }

  float shift_w = 0.0, shift_h = 0.0, scale_w = 0., scale_h = 0.0;
  scale_w = float(input_w) / resize_w;
  scale_h = float(input_h) / resize_h;
  int step = output_w * output_h;
  for (int i = 0; i < model_param.anchor_num; ++i) {
    int data_index =
        i * (output_c / model_param.anchor_num) * output_h * output_w;
    for (int h = 0; h < output_h; ++h)
      for (int w = 0; w < output_w; ++w) {
        BBox bbox;
        bbox.x = (w + Sigmoid(output_data[data_index + h * output_w + w])) /
                 output_w;
        bbox.y =
            (h + Sigmoid(output_data[data_index + h * output_w + w + step])) /
            output_h;
        bbox.w =
            exp(output_data[data_index + h * output_w + w + 2 * step]) *
            model_param.biases[out_index * model_param.anchor_num * 2 + i * 2] /
            output_w;
        bbox.h =
            exp(output_data[data_index + h * output_w + w + 3 * step]) *
            model_param
                .biases[out_index * model_param.anchor_num * 2 + i * 2 + 1] /
            output_h;

        bbox.x = std::max((bbox.x - shift_w - bbox.w / 2) * scale_w * img_width,
                          0.0f);
        bbox.y = std::max(
            (bbox.y - shift_h - bbox.h / 2) * scale_h * img_height, 0.0f);
        bbox.w *= (scale_w * img_width);
        bbox.h *= (scale_h * img_height);
        bbox.w = std::min(bbox.w, img_width - bbox.x - 1);
        bbox.h = std::min(bbox.h, img_height - bbox.y - 1);

        if (bbox.w <= 0 or bbox.h <= 0) {
          continue;
        }

        float confidence =
            Sigmoid(output_data[data_index + h * output_w + w + 4 * step]);
        float classScore;
        ClassIndexAndScore(output_data,
                           data_index + h * output_w + w + 5 * step, step,
                           model_param.classes, bbox.category, classScore);
        bbox.score = confidence * classScore;
        if (bbox.score > model_param.score_thresh) {
          bboxes.push_back(bbox);
        }
      }
  }
}

static float Sigmoid(float x) { return 1. / (1. + exp(-x)); }

static float overlap(BBox bbox1, BBox bbox2) {
  float left = std::max(bbox1.x, bbox2.x);
  float right = std::min(bbox1.x + bbox1.w, bbox2.x + bbox2.w);
  float top = std::max(bbox1.y, bbox2.y);
  float bottom = std::min(bbox1.y + bbox1.h, bbox2.y + bbox2.h);
  if (left >= right or top >= bottom) {
    return 0;
  } else {
    float inter_area = (right - left) * (bottom - top);
    float union_area = bbox1.w * bbox1.h + bbox2.w * bbox2.h - inter_area;
    return inter_area / union_area;
  }
}

static void ClassIndexAndScore(float *input, int index, int step, int classes,
                               int &classIdx, float &classScore) {
  float sum = 0;
  float large = input[index];
  int classIndex = 0;
  for (int i = 0; i < classes; ++i) {
    if (input[i * step + index] > large) large = input[i * step + index];
  }
  for (int i = 0; i < classes; ++i) {
    float e = exp(input[i * step + index] - large);
    sum += e;
    input[i * step + index] = e;
  }
  for (int i = 0; i < classes; ++i) {
    input[i * step + index] = input[i * step + index] / sum;
  }
  large = input[index];
  classIndex = 0;
  for (int i = 0; i < classes; ++i) {
    if (input[i * step + index] > large) {
      large = input[i * step + index];
      classIndex = i;
    }
  }
  classIdx = classIndex;
  classScore = large;
}

static void ApplyNMSV2(std::vector<BBox> &bboxes, std::vector<BBox> &outBBoxes,
                       float nms) {
  for (auto &curBox : bboxes) {
    bool isOverlap = false;
    for (auto &outBox : outBBoxes) {
      if (overlap(curBox, outBox) >= nms) {
        isOverlap = true;
        break;
      }
    }
    if (!isOverlap) {
      outBBoxes.push_back(curBox);
    }
  }
}

MODELBOX_FLOWUNIT(CarYoloboxFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput(
      modelbox::FlowUnitInput("layer15-conv", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitInput(
      modelbox::FlowUnitInput("layer22-conv", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitOutput(modelbox::FlowUnitOutput("Out_1", modelbox::DEVICE_TYPE));
  desc.SetFlowType(modelbox::NORMAL);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
