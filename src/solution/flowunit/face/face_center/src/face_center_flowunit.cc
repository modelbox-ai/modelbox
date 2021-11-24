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


#include "face_center_flowunit.h"

#include "securec.h"

#include <cmath>
#include <opencv2/core.hpp>
#include <vector>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

using std::map;
using std::vector;
using namespace modelbox;

static void ApplyNMSV2(std::vector<CenterFaces> &input,
                       std::vector<CenterFaces> &output, float nms);

FaceCenterFlowUnit::FaceCenterFlowUnit(){};
FaceCenterFlowUnit::~FaceCenterFlowUnit(){};

modelbox::Status FaceCenterFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  MBLOG_DEBUG << "centerv2box open : begin to get parameters";
  original_img_w_ = opts->GetUint32("image_width", ORIGINAL_IMAGE_W_DEFAULT);
  original_img_h_ = opts->GetUint32("image_height", ORIGINAL_IMAGE_H_DEFAULT);
  if (original_img_w_ <= 0 || original_img_h_ <= 0) {
    auto errMsg = "image width or height is invalid.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_BADCONF, errMsg};
  }

  net_input_w_ = opts->GetUint32("input_width", NET_INPUT_W_DEFAULT);
  net_input_h_ = opts->GetUint32("input_height", NET_INPUT_H_DEFAULT);
  if (net_input_w_ <= 0 || net_input_h_ <= 0) {
    auto errMsg = "input width or height is invalid.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_BADCONF, errMsg};
  }
  auto status = InitCenterFlowunit();
  if (!status) {
    return status;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status FaceCenterFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_DEBUG << "process image with centervbox";
  // checkout input batch_size and shape
  std::vector<std::shared_ptr<BufferList>> infer_layers;
  auto status = GetInferLayers(data_ctx, infer_layers);
  if (!status) {
    return status;
  }

  // Get the face fboxes of all pictures in the batch
  vector<vector<CenterFaces>> out_bboxes;
  for (size_t batch_idx = 0; batch_idx < infer_layers[0]->Size(); ++batch_idx) {
    auto one_batch_bboxes = ComputerBox(infer_layers, batch_idx);
    out_bboxes.push_back(one_batch_bboxes);
  }
  // Send face boxes to draw flowunit
  status = SendBox(data_ctx, out_bboxes);
  if (!status) {
    return status;
  }
  // Send face keyponits to face align flowunit
  status = SendKeyPoints(data_ctx, out_bboxes);
  if (!status) {
    return status;
  }

  return modelbox::STATUS_OK;
}

bool FaceCenterFlowUnit::CheckShape(ImageShape shape,
                                    const std::vector<size_t> &input) {
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

modelbox::Status FaceCenterFlowUnit::GetInferLayers(
    std::shared_ptr<modelbox::DataContext> data_ctx,
    std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers) {
  size_t batch_size = 0;
  for (size_t i = 0; i < CENTER_UNIT_IN_NAME.size(); ++i) {
    // check batch size, all input batch size is same
    std::shared_ptr<BufferList> layer = data_ctx->Input(CENTER_UNIT_IN_NAME[i]);
    auto cur_batch_size = layer->Size();
    if (cur_batch_size <= 0 ||
        (batch_size != 0 && cur_batch_size != batch_size)) {
      auto errMsg =
          "infer layer is invalid. batch_size:" + std::to_string(batch_size) +
          " layer_name:" + CENTER_UNIT_IN_NAME[i] +
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

vector<CenterFaces> FaceCenterFlowUnit::ComputerBox(
    std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers,
    int batch_idx) {
  // one batch
  vector<CenterFaces> results;
  vector<CenterFaces> all_layer_bboxes;
  vector<CenterFaces> layer_bboxes;

  computeCenterBBox(infer_layers, layer_bboxes, batch_idx);
  all_layer_bboxes.insert(all_layer_bboxes.end(), layer_bboxes.begin(),
                          layer_bboxes.end());

  // nms
  std::sort(all_layer_bboxes.begin(), all_layer_bboxes.end(),
            [](const CenterFaces &bbox1, const CenterFaces &bbox2) {
              return bbox1.score > bbox2.score;
            });
  ApplyNMSV2(all_layer_bboxes, results, NMS_THRESHOLD);

  // log
  for (auto box : results) {
    MBLOG_DEBUG << "batch idx:" << batch_idx << " [" << box.pt1.x << ","
                << box.pt1.y << "," << box.pt2.x << "," << box.pt2.y << "] "
                << box.score;
  }

  return results;
}

modelbox::Status FaceCenterFlowUnit::SendBox(
    std::shared_ptr<modelbox::DataContext> data_ctx,
    const std::vector<std::vector<CenterFaces>> &out_bboxes) {
  // out boxes output shape
  std::vector<size_t> shape;
  for (auto &boxes : out_bboxes) {
    shape.emplace_back(boxes.size() * sizeof(CenterFaces));
  }
  auto output_bufs = data_ctx->Output(CENTER_UNIT_OUT_NAME[0]);
  auto ret = output_bufs->Build(shape);
  if (!ret) {
    auto errMsg = "build face boxes output failed in face center unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  // output
  for (size_t i = 0; i < shape.size(); ++i) {
    MBLOG_DEBUG << "batch:" << i << ", size:" << out_bboxes[i].size();
    auto out_data = (CenterFaces *)(*output_bufs)[i]->MutableData();
    for (auto &boxes : out_bboxes[i]) {
      *out_data = boxes;
      out_data++;
    }
  }
  return modelbox::STATUS_OK;
}

modelbox::Status FaceCenterFlowUnit::SendKeyPoints(
    std::shared_ptr<modelbox::DataContext> data_ctx,
    const std::vector<std::vector<CenterFaces>> &out_bboxes) {
  errno_t err;
  // out boxes output shape
  std::vector<size_t> shape;
  for (auto &boxes : out_bboxes) {
    shape.emplace_back(boxes.size() * sizeof(CenterKeyPoints));
  }
  auto output_bufs = data_ctx->Output(CENTER_UNIT_OUT_NAME[1]);
  auto ret = output_bufs->Build(shape);
  if (!ret) {
    auto errMsg = "build keypoints output failed in face center unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  // output
  for (size_t i = 0; i < shape.size(); ++i) {
    MBLOG_DEBUG << "batch:" << i << ", size:" << out_bboxes[i].size();
    auto out_data = (CenterKeyPoints *)(*output_bufs)[i]->MutableData();
    for (auto &boxes : out_bboxes[i]) {
      err = memcpy_s(out_data, sizeof(CenterKeyPoints), boxes.kps,
                     5 * sizeof(CoordinatePoints));
      if (err != 0) {
        auto errMsg =
            "Copying key points into output port in face center unit failed";
        MBLOG_ERROR << errMsg;
        return {modelbox::STATUS_FAULT, errMsg};
      }
      out_data++;
    }
  }
  return modelbox::STATUS_OK;
}

void FaceCenterFlowUnit::computeCenterBBox(
    std::vector<std::shared_ptr<modelbox::BufferList>> &infer_layers,
    vector<CenterFaces> &bboxes, int batch_idx) {
  /*
   *   ----------------------------------------------------------------------------
   *   |  score    | x_offset | y_offset |   w   |   h   |           kps |
   *   ----------------------------------------------------------------------------
   *   |center_data|     offset_data     |  matrix_data  |          kps_data |
   *   ----------------------------------------------------------------------------
   *   |   L       |         2L          |      2L       |            10L |
   *   ----------------------------------------------------------------------------
   *   |  layer[0] |       layer[1]      |    layer[2]   |          layer[3] |
   *   ----------------------------------------------------------------------------
   *   L = feature_size_
   */
  float x1 = 0;
  float x2 = 0;
  float y1 = 0;
  float y2 = 0;
  float score = 0;
  if (infer_layers.size() != OUTPUT_CHANNEL) {
    return;
  }

  float *center_data = (float *)(infer_layers[0]->At(batch_idx)->ConstData());
  float *offset_data = (float *)(infer_layers[1]->At(batch_idx)->ConstData());
  float *matrix_data = (float *)(infer_layers[2]->At(batch_idx)->ConstData());
  float *kps_data = (float *)(infer_layers[3]->At(batch_idx)->ConstData());

  CenterFaces face;
  for (int i = 0; i < net_output_h_; i++) {    // y
    for (int j = 0; j < net_output_w_; j++) {  // x
      int index = i * net_output_w_ + j;
      if (*(center_data + index) > SCORE_THRESHOLD) {
        x1 = (float(j) + *(offset_data + index)) * 4 -
             *(matrix_data + index) * 2;
        y1 = (float(i) + *(offset_data + index + feature_size_)) * 4 -
             *(matrix_data + index + feature_size_) * 2;
        x2 = (float(j) + *(offset_data + index)) * 4 +
             *(matrix_data + index) * 2;
        y2 = (float(i) + *(offset_data + index + feature_size_)) * 4 +
             *(matrix_data + index + feature_size_) * 2;
        score = *(center_data + index);
        face.pt1.x = (x1 - w_padding_) / img_scale_;
        face.pt1.y = (y1 - h_padding_) / img_scale_;
        face.pt2.x = (x2 - w_padding_) / img_scale_;
        face.pt2.y = (y2 - h_padding_) / img_scale_;
        face.score = score;
        for (int k = 0; k < 5; k++) {
          face.kps[k].x =
              ((float(j) + *(kps_data + (index + 2 * k * feature_size_))) * 4 -
               w_padding_) /
              img_scale_;
          face.kps[k].y =
              ((float(i) +
                *(kps_data + (index + (2 * k + 1) * feature_size_))) *
                   4 -
               h_padding_) /
              img_scale_;
        }
        bboxes.push_back(face);
      }
    }
  }
}

static void ApplyNMSV2(std::vector<CenterFaces> &bboxes,
                       std::vector<CenterFaces> &outBBoxes, float nms) {
  int box_num = int(bboxes.size());
  std::vector<CenterFaces> output;
  std::vector<int> suppressed(box_num, 0);
  for (int i = 0; i < box_num; i++) {
    if (suppressed[i]) {
      continue;
    }
    outBBoxes.push_back(bboxes[i]);

    for (int j = i + 1; j < box_num; j++) {
      // skip if suppressed already
      if (suppressed[j]) {
        continue;
      }
      // compute cross region
      float ix1 = std::max(bboxes[i].pt1.x, bboxes[j].pt1.x);
      float iy1 = std::max(bboxes[i].pt1.y, bboxes[j].pt1.y);
      float ix2 = std::min(bboxes[i].pt2.x, bboxes[j].pt2.x);
      float iy2 = std::min(bboxes[i].pt2.y, bboxes[j].pt2.y);
      float ih = iy2 - iy1 + 1;
      float iw = ix2 - ix1 + 1;

      // continue if no cross
      if (ih <= 0 || iw <= 0) {
        continue;
      }

      // iou suppressed
      float area1 = (bboxes[i].pt2.x - bboxes[i].pt1.x + 1) *
                    (bboxes[i].pt2.y - bboxes[i].pt1.y + 1);
      float area2 = (bboxes[j].pt2.x - bboxes[j].pt1.x + 1) *
                    (bboxes[j].pt2.y - bboxes[j].pt1.y + 1);
      float intersection = ih * iw;
      float union_area = area1 + area2 - intersection;
      if (union_area == 0) {
        continue;
      }
      float iou = intersection / union_area;
      if (iou > nms) {
        suppressed[j] = 1;
      }
    }
  }
}

// Calculation of the pre-parameters required to initialize the stream unit
modelbox::Status FaceCenterFlowUnit::InitCenterFlowunit(void) {
  // Calculate the size of the output feature map
  net_output_h_ = net_input_h_ / 4;
  net_output_w_ = net_input_w_ / 4;
  feature_size_ = net_output_w_ * net_output_h_;
  out_shape_ = {{.w = net_output_w_, .h = net_output_h_, .c = 1, .n = 1},
                {.w = net_output_w_, .h = net_output_h_, .c = 2, .n = 1},
                {.w = net_output_w_, .h = net_output_h_, .c = 2, .n = 1},
                {.w = net_output_w_, .h = net_output_h_, .c = 10, .n = 1}};

  // Calculate the padding value according to the resize method
  img_scale_ = cv::min(float(net_input_w_) / original_img_w_,
                       float(net_input_h_) / original_img_h_);
  if (img_scale_ == 0) {
    auto errMsg =
        "padding size is wrong, please check image input width or height";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  auto scaleSize =
      cv::Size(original_img_w_ * img_scale_, original_img_h_ * img_scale_);
  w_padding_ = (net_input_w_ - scaleSize.width) / 2;
  h_padding_ = (net_input_h_ - scaleSize.height) / 2;

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(FaceCenterFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  for (auto port : CENTER_UNIT_IN_NAME) {
    desc.AddFlowUnitInput(modelbox::FlowUnitInput(port, modelbox::DEVICE_TYPE));
  }

  for (auto port : CENTER_UNIT_OUT_NAME) {
    desc.AddFlowUnitOutput(modelbox::FlowUnitOutput(port, modelbox::DEVICE_TYPE));
  }
  desc.SetFlowType(modelbox::NORMAL);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "original_img_w_", "int", true, std::to_string(ORIGINAL_IMAGE_W_DEFAULT),
      "the original image width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "original_img_h_", "int", true, std::to_string(ORIGINAL_IMAGE_H_DEFAULT),
      "the original image height"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_input_w_", "int", true, std::to_string(NET_INPUT_W_DEFAULT),
      "the net input width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_input_h_", "int", true, std::to_string(NET_INPUT_H_DEFAULT),
      "the net input height"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}