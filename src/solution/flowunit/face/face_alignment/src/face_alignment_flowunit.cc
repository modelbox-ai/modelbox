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


#include "face_alignment_flowunit.h"

#include <securec.h>

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

using std::map;
using std::vector;
using namespace modelbox;

FaceAlignmentFlowUnit::FaceAlignmentFlowUnit(){};
FaceAlignmentFlowUnit::~FaceAlignmentFlowUnit(){};

modelbox::Status FaceAlignmentFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  net_input_w_ = opts->GetUint32("net_width", NET_INPUT_W_DEFAULT);
  net_input_h_ = opts->GetUint32("net_height", NET_INPUT_H_DEFAULT);
  if (net_input_w_ <= 0 || net_input_h_ <= 0) {
    auto errMsg = "input width or height is invalid.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_BADCONF, errMsg};
  }
  InitRadiationMatrix();

  return modelbox::STATUS_OK;
}

modelbox::Status FaceAlignmentFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  MBLOG_DEBUG << "process alignment face";
  // face image
  auto input1_bufs = data_ctx->Input(ALIGN_UNIT_IN_NAME[0]);
  if (input1_bufs->Size() <= 0) {
    auto errMsg = "In_img batch is " + std::to_string(input1_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  // face keypoints
  auto input2_bufs = data_ctx->Input(ALIGN_UNIT_IN_NAME[1]);
  if (input2_bufs->Size() <= 0) {
    auto errMsg = "In_kps batch is " + std::to_string(input2_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  auto output_bufs = data_ctx->Output(ALIGN_UNIT_OUT_NAME[0]);
  errno_t err;
  // Check port size
  auto status = CheckChannelSize(input1_bufs->Size(), input2_bufs->Size());
  if (!status) {
    return status;
  }

  size_t channel = RGB_CHANNELS;
  std::vector<size_t> tensor_shape;
  for (size_t i = 0; i < input2_bufs->Size(); ++i) {
    // get face number in each img
    size_t face_num = input2_bufs->At(i)->GetBytes() / sizeof(FacePoints);
    tensor_shape.emplace_back(net_input_w_ * net_input_h_ * channel *
                              sizeof(float) * face_num);
  }
  auto ret = output_bufs->Build(tensor_shape);
  if (!ret) {
    auto errMsg = "build output failed in face align unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  for (size_t i = 0; i < input2_bufs->Size(); ++i) {
    // get face number in one img
    if (input2_bufs->At(i)->GetBytes() == 0) {
      continue;
    }

    size_t face_num = input2_bufs->At(i)->GetBytes() / sizeof(FacePoints);
    MBLOG_INFO << "face num: " << face_num;
    std::vector<std::shared_ptr<FacePoints>> faces;
    for (size_t j = 0; j < face_num; ++j) {
      std::shared_ptr<FacePoints> f = std::make_shared<FacePoints>();
      err = memcpy_s(f.get(), sizeof(FacePoints),
                     (const char *)(input2_bufs->ConstBufferData(i)) +
                         (sizeof(FacePoints) * j),
                     sizeof(FacePoints));
      if (err != 0) {
        auto errMsg = "Copying face points in face align unit failed";
        MBLOG_ERROR << errMsg;
        return {modelbox::STATUS_FAULT, errMsg};
      }
      faces.push_back(f);
    }
    // get images
    int32_t width = 0, height = 0, channel = 0, rate_den = 0, rate_num = 0;
    input1_bufs->At(i)->Get("width", width);
    input1_bufs->At(i)->Get("height", height);
    input1_bufs->At(i)->Get("channel", channel);
    input1_bufs->At(i)->Get("rate_den", rate_den);
    input1_bufs->At(i)->Get("rate_num", rate_num);

    cv::Mat image;
    cv::Mat img(height, width, CV_8UC3);
    err = memcpy_s(img.data, img.total() * img.elemSize(),
                   input1_bufs->ConstBufferData(i),
                   input1_bufs->At(i)->GetBytes());
    if (err != 0) {
      auto errMsg = "Copying origin image in face align unit failed";
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }
    img.convertTo(image, CV_32FC3, 1.);
    // output data
    auto output_data = static_cast<float *>((*output_bufs)[i]->MutableData());
    cv::Mat rgb_img;

    for (auto &face : faces) {
      EachFaceAlignment(image, face, rgb_img);
      std::vector<cv::Mat> input_channels(channel);
      cv::split(rgb_img, input_channels);

      // normalize
      int channelLength = net_input_h_ * net_input_w_;
      std::vector<float> result(channelLength * channel);
      auto data = result.data();

      for (int i = 0; i < channel; ++i) {
        cv::Mat normed_channel =
            (input_channels[i] - FACE_MEAN[i]) * FACE_VARIANCE[i];
        err = memcpy_s(data, channelLength * sizeof(float), normed_channel.data,
                       channelLength * sizeof(float));
        if (err != 0) {
          auto errMsg =
              "Copying image into transpose image in face align unit failed";
          MBLOG_ERROR << errMsg;
          return {modelbox::STATUS_FAULT, errMsg};
        }
        data += channelLength;
      }
      err = memcpy_s(output_data, channelLength * channel * sizeof(float),
                     result.data(), channelLength * channel * sizeof(float));
      if (err != 0) {
        auto errMsg =
            "Copying image into output port in face align unit failed";
        MBLOG_ERROR << errMsg;
        return {modelbox::STATUS_FAULT, errMsg};
      }
      output_data += channelLength * channel;
    }

    output_bufs->At(i)->Set("width", int32_t(net_input_w_));
    output_bufs->At(i)->Set("height", int32_t(net_input_h_));
    output_bufs->At(i)->Set("channel", channel);
    output_bufs->At(i)->Set("pix_fmt", std::string("rgb"));
    output_bufs->At(i)->Set("rate_den", rate_den);
    output_bufs->At(i)->Set("rate_num", rate_num);
  }

  MBLOG_INFO << "face align process finished";
  return modelbox::STATUS_OK;
}

modelbox::Status FaceAlignmentFlowUnit::CheckChannelSize(int port1Size,
                                                       int port2Size) {
  auto errMsg = "In_1 batch is " + std::to_string(port1Size) +
                "In_2 batch is " + std::to_string(port2Size);

  if (port1Size <= 0 || port2Size <= 0) {
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  if (port1Size != port2Size) {
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  return modelbox::STATUS_OK;
}

void FaceAlignmentFlowUnit::InitRadiationMatrix(void) {
  std::vector<cv::Point> pts;
  pts.push_back(cv::Point2f(87.5305, 118.1630));   // left eye
  pts.push_back(cv::Point2f(168.0727, 117.7175));  // right eye
  pts.push_back(cv::Point2f(128.0576, 163.9694));  // noise
  pts.push_back(cv::Point2f(94.9698, 211.1211));   // left mouth
  pts.push_back(cv::Point2f(161.6683, 210.7522));  // right mouth
  cv::Mat A(2 * pts.size(), 4, CV_32FC1);
  for (int i = 0; i < int(pts.size()); i++) {
    const cv::Point2f &p = pts[i];
    A.at<float>(2 * i, 0) = p.x;
    A.at<float>(2 * i, 1) = -p.y;
    A.at<float>(2 * i, 2) = 1.f;
    A.at<float>(2 * i, 3) = 0.f;
    A.at<float>(2 * i + 1, 0) = p.y;
    A.at<float>(2 * i + 1, 1) = p.x;
    A.at<float>(2 * i + 1, 2) = 0.f;
    A.at<float>(2 * i + 1, 3) = 1.f;
  }

  cv::Mat At;
  cv::transpose(A, At);
  cv::Mat AtA = At * A;
  cv::Mat AtA_inv;
  cv::invert(AtA, AtA_inv);
  matrixA_ = AtA_inv * At;

  return;
}

void FaceAlignmentFlowUnit::EachFaceAlignment(
    const cv::Mat &image, std::shared_ptr<FacePoints> &keypoints,
    cv::Mat &aligned_image) {
  cv::Mat B(2 * KPS_NUM, 1, CV_32FC1, (void *)keypoints.get());
  cv::Mat matrixAB = matrixA_ * B;
  float a = matrixAB.at<float>(0, 0);
  float b = matrixAB.at<float>(1, 0);
  float c = matrixAB.at<float>(2, 0);
  float d = matrixAB.at<float>(3, 0);

  cv::Mat tempMatrix(2, 3, CV_32FC1);
  tempMatrix.at<float>(0, 0) = a;
  tempMatrix.at<float>(0, 1) = -b;
  tempMatrix.at<float>(0, 2) = c;
  tempMatrix.at<float>(1, 0) = b;
  tempMatrix.at<float>(1, 1) = a;
  tempMatrix.at<float>(1, 2) = d;
  cv::Size output_size(net_input_w_, net_input_h_);
  cv::warpAffine(image, aligned_image, tempMatrix, output_size,
                 cv::WARP_INVERSE_MAP | cv::INTER_LINEAR);

  return;
}

MODELBOX_FLOWUNIT(FaceAlignmentFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  for (auto port : ALIGN_UNIT_IN_NAME) {
    desc.AddFlowUnitInput(modelbox::FlowUnitInput(port, modelbox::DEVICE_TYPE));
  }
  for (auto port : ALIGN_UNIT_OUT_NAME) {
    desc.AddFlowUnitOutput(modelbox::FlowUnitOutput(port, modelbox::DEVICE_TYPE));
  }
  desc.SetFlowType(modelbox::NORMAL);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
