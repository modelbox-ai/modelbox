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


#include "face_draw_flowunit.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

FaceDrawAllFlowUnit::FaceDrawAllFlowUnit(){};
FaceDrawAllFlowUnit::~FaceDrawAllFlowUnit(){};

std::vector<std::string> kCVResizeMethod = {"all", "max"};

modelbox::Status FaceDrawAllFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  method_ = opts->GetString("method", "all");
  if (method_ != "all" && method_ != "max") {
    auto errMsg = "Drawing mode is not configured or invalid.";
    MBLOG_ERROR << errMsg << "method : " << method_;
    return {modelbox::STATUS_BADCONF, errMsg};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status FaceDrawAllFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status FaceDrawAllFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_INFO << "draw face box and face expression";
  // face boxes
  auto input1_bufs = ctx->Input(DRAW_UNIT_IN_NAME[0]);
  if (input1_bufs->Size() <= 0) {
    auto errMsg = "In_1 batch in face draw unit is " +
                  std::to_string(input1_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  // recive origin image
  auto input2_bufs = ctx->Input(DRAW_UNIT_IN_NAME[1]);
  if (input2_bufs->Size() <= 0) {
    auto errMsg = "In_2 batch in face draw unit is " +
                  std::to_string(input2_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  // face expression mode
  auto input3_bufs = ctx->Input(DRAW_UNIT_IN_NAME[2]);
  if (input3_bufs->Size() <= 0) {
    auto errMsg = "In_3 batch in face draw unit is " +
                  std::to_string(input3_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  // output image
  auto output_bufs = ctx->Output(DRAW_UNIT_OUT_NAME[0]);
  // Check port size
  auto status = CheckChannelSize(input1_bufs->Size(), input2_bufs->Size(),
                                 input3_bufs->Size());
  if (!status) {
    return status;
  }

  std::vector<size_t> shape;
  // The output memory size is the same as the original image memory size
  for (size_t i = 0; i < input2_bufs->Size(); ++i) {
    shape.emplace_back(input2_bufs->At(i)->GetBytes());
  }
  auto ret = output_bufs->Build(shape);
  if (!ret) {
    auto errMsg = "build output failed in face draw unit";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }
  errno_t err;
  for (size_t i = 0; i < input1_bufs->Size(); ++i) {
    // get images
    int32_t width = 0, height = 0, channel = 0, rate_den = 0, rate_num = 0;
    std::string pix_fmt;
    input2_bufs->At(i)->Get("width", width);
    input2_bufs->At(i)->Get("height", height);
    input2_bufs->At(i)->Get("channel", channel);
    input2_bufs->At(i)->Get("pix_fmt", pix_fmt);
    input2_bufs->At(i)->Get("rate_den", rate_den);
    input2_bufs->At(i)->Get("rate_num", rate_num);

    cv::Mat image(height, width, CV_8UC3);
    err = memcpy_s(image.data, image.total() * image.elemSize(),
                   input2_bufs->ConstBufferData(i),
                   input2_bufs->At(i)->GetBytes());
    if (err != 0) {
      auto errMsg =
          "Copying origin image from In_2 port in face draw unit failed.";
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }

    MBLOG_INFO << "end get images";

    if (input1_bufs->At(i)->GetBytes() == 0) {
      // output data
      auto output_data = output_bufs->MutableBufferData(i);
      err = memcpy_s(output_data, output_bufs->At(i)->GetBytes(), image.data,
                     image.total() * image.elemSize());
      if (err != 0) {
        auto errMsg =
            "Copying output image to output port in face draw unit failed.";
        MBLOG_ERROR << errMsg;
        return {modelbox::STATUS_FAULT, errMsg};
      }

      output_bufs->At(i)->Set("width", width);
      output_bufs->At(i)->Set("height", height);
      output_bufs->At(i)->Set("channel", channel);
      output_bufs->At(i)->Set("pix_fmt", pix_fmt);
      output_bufs->At(i)->Set("rate_den", rate_den);
      output_bufs->At(i)->Set("rate_num", rate_num);
      continue;
    }
    // get bboxes
    size_t num_bboxes = input1_bufs->At(i)->GetBytes() / sizeof(CenterFaces);
    MBLOG_INFO << "num_bboxes: " << num_bboxes;
    std::vector<std::shared_ptr<CenterFaces>> bboxs;
    for (size_t j = 0; j < num_bboxes; ++j) {
      std::shared_ptr<CenterFaces> b = std::make_shared<CenterFaces>();
      err = memcpy_s(b.get(), sizeof(CenterFaces),
                     (const char *)(input1_bufs->ConstBufferData(i)) +
                         (sizeof(CenterFaces) * j),
                     sizeof(CenterFaces));
      if (err != 0) {
        auto errMsg =
            "Copying face keypoints from In_1 port in face draw unit failed.";
        MBLOG_ERROR << errMsg;
        return {modelbox::STATUS_FAULT, errMsg};
      }
      bboxs.push_back(b);
    }
    // get faces
    size_t face_num =
        input3_bufs->At(i)->GetBytes() / sizeof(FaceExpressionScore);
    MBLOG_INFO << "num_faces: " << face_num;
    std::vector<std::shared_ptr<FaceExpressionScore>> faces;
    for (size_t j = 0; j < face_num; ++j) {
      std::shared_ptr<FaceExpressionScore> f =
          std::make_shared<FaceExpressionScore>();
      err = memcpy_s(f.get(), sizeof(FaceExpressionScore),
                     (const char *)(input3_bufs->ConstBufferData(i)) +
                         (sizeof(FaceExpressionScore) * j),
                     sizeof(FaceExpressionScore));
      if (err != 0) {
        auto errMsg =
            "Copying expression score from In_3 port in face draw unit failed.";
        MBLOG_ERROR << errMsg;
        return {modelbox::STATUS_FAULT, errMsg};
      }
      faces.push_back(f);
    }

    int face_idx = 0;
    for (auto &box : bboxs) {
      cv::rectangle(image, cv::Point(box->pt1.x, box->pt1.y),
                    cv::Point(box->pt2.x, box->pt2.y), cv::Scalar(255, 0, 0), 4,
                    8, 0);
      if (method_ == "all") {
        DrawAllExpressionScore(image, faces[face_idx], box);
      }
      if (method_ == "max") {
        DrawMaxExpressionScore(image, faces[face_idx], box);
      }
      for (int k = 0; k < 5; k++) {
        cv::circle(image, cv::Point(box->kps[k].x, box->kps[k].y), 2,
                   cv::Scalar(255, 0, 0), 4);
      }
    }

    // output data
    auto output_data = output_bufs->MutableBufferData(i);
    err = memcpy_s(output_data, output_bufs->At(i)->GetBytes(), image.data,
                   image.total() * image.elemSize());
    if (err != 0) {
      auto errMsg =
          "Copying output image to output port in face draw unit failed.";
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }
    output_bufs->At(i)->Set("width", width);
    output_bufs->At(i)->Set("height", height);
    output_bufs->At(i)->Set("channel", channel);
    output_bufs->At(i)->Set("pix_fmt", pix_fmt);
    output_bufs->At(i)->Set("rate_den", rate_den);
    output_bufs->At(i)->Set("rate_num", rate_num);
  }
  return modelbox::STATUS_OK;
}

void FaceDrawAllFlowUnit::DrawAllExpressionScore(
    cv::Mat image, std::shared_ptr<FaceExpressionScore> face,
    std::shared_ptr<CenterFaces> box) {
  std::string surprise_score =
      "surprise_score:" + std::to_string(face->surprise_score);
  std::string fear_score = "fear_score:" + std::to_string(face->fear_score);
  std::string disgust_score =
      "disgust_score:" + std::to_string(face->disgust_score);
  std::string happy_score = "happy_score:" + std::to_string(face->happy_score);
  std::string sad_score = "sad_score:" + std::to_string(face->sad_score);
  std::string angry_score = "angry_score:" + std::to_string(face->angry_score);
  std::string neutral_score =
      "neutral_score:" + std::to_string(face->neutral_score);

  int space = (box->pt2.y - box->pt1.y) / 6;
  cv::putText(image, surprise_score, cv::Point(box->pt2.x, box->pt1.y),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 8, false);
  cv::putText(image, fear_score, cv::Point(box->pt2.x, box->pt1.y + space),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 8, false);
  cv::putText(image, disgust_score,
              cv::Point(box->pt2.x, box->pt1.y + space * 2),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 8, false);
  cv::putText(image, happy_score, cv::Point(box->pt2.x, box->pt1.y + space * 3),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 8, false);
  cv::putText(image, sad_score, cv::Point(box->pt2.x, box->pt1.y + space * 4),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 8, false);
  cv::putText(image, angry_score, cv::Point(box->pt2.x, box->pt1.y + space * 5),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 8, false);
  cv::putText(image, neutral_score, cv::Point(box->pt2.x, box->pt2.y),
              cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1, 8, false);

  return;
}

void FaceDrawAllFlowUnit::DrawMaxExpressionScore(
    cv::Mat image, std::shared_ptr<FaceExpressionScore> face,
    std::shared_ptr<CenterFaces> box) {
  float maxScore = 0;
  int maxIdx = 0;
  float score[7];
  score[0] = face->surprise_score;
  score[1] = face->fear_score;
  score[2] = face->disgust_score;
  score[3] = face->happy_score;
  score[4] = face->sad_score;
  score[5] = face->angry_score;
  score[6] = face->neutral_score;
  for (int i = 0; i < 7; ++i) {
    if (score[i] > maxScore) {
      maxScore = score[i];
      maxIdx = i;
    }
  }
  std::vector<std::string> expression = {
      "surprise :", "fear :",  "disgust :", "happy :",
      "sad :",      "angry :", "neutral :"};

  std::string expression_score =
      expression[maxIdx] + std::to_string(score[maxIdx]);

  cv::putText(image, expression_score, cv::Point(box->pt2.x, box->pt1.y),
              cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 0, 0), 1, 8, false);

  return;
}

modelbox::Status FaceDrawAllFlowUnit::CheckChannelSize(int port1Size,
                                                     int port2Size,
                                                     int port3Size) {
  auto errMsg = "In_1 batch is " + std::to_string(port1Size) +
                "In_2 batch is " + std::to_string(port2Size) +
                "In_3 batch is " + std::to_string(port3Size);

  if (port1Size != port2Size || port1Size != port3Size) {
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(FaceDrawAllFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  for (auto port : DRAW_UNIT_IN_NAME) {
    desc.AddFlowUnitInput(modelbox::FlowUnitInput(port, modelbox::DEVICE_TYPE));
  }

  for (auto port : DRAW_UNIT_OUT_NAME) {
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
