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


#include "draw_bbox_mot_flowunit.h"

#include <securec.h>

#include "modelbox/device/cpu/device_cpu.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

DrawBBoxMOTFlowUnit::DrawBBoxMOTFlowUnit(){};
DrawBBoxMOTFlowUnit::~DrawBBoxMOTFlowUnit(){};

modelbox::Status DrawBBoxMOTFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  return modelbox::STATUS_OK;
}

modelbox::Status DrawBBoxMOTFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_DEBUG << "process image draw bbox on cpu";
  auto input1_bufs = ctx->Input("In_1");
  if (input1_bufs->Size() <= 0) {
    auto errMsg = "In_1 images batch is " + std::to_string(input1_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto input2_bufs = ctx->Input("In_2");
  if (input2_bufs->Size() <= 0) {
    auto errMsg = "In_2 batch is " + std::to_string(input2_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  if (input1_bufs->Size() != input2_bufs->Size()) {
    auto errMsg = "In_1 batch is not match In_2 batch. In_1 is " +
                  std::to_string(input1_bufs->Size()) + ",In_2 is " +
                  std::to_string(input2_bufs->Size());
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto output_bufs = ctx->Output("Out_1");

  std::vector<size_t> shape;
  for (size_t i = 0; i < input2_bufs->Size(); ++i) {
    shape.emplace_back(input2_bufs->At(i)->GetBytes());
  }

  auto shape_ret = output_bufs->Build(shape);
  if (!shape_ret) {
    MBLOG_ERROR << "draw_bbox_mot : get output memory failed.";
    return modelbox::STATUS_NOMEM;
  }

  MBLOG_DEBUG << "begin process batch";

  for (size_t i = 0; i < input1_bufs->Size(); ++i) {
    size_t num_stracks = input1_bufs->At(i)->GetBytes() / sizeof(TrackResult);

    std::vector<std::shared_ptr<TrackResult>> output_stracks;
    for (size_t j = 0; j < num_stracks; ++j) {
      std::shared_ptr<TrackResult> strack = std::make_shared<TrackResult>();
      auto strack_ret =
          memcpy_s(strack.get(), sizeof(TrackResult),
                   (const char *)(input1_bufs->ConstBufferData(i)) +
                       (sizeof(TrackResult) * j),
                   sizeof(TrackResult));
      if (EOK != strack_ret) {
        MBLOG_ERROR << "Cpu draw_bbox_mot failed, strack_ret " << strack_ret;
        return modelbox::STATUS_FAULT;
      }

      output_stracks.push_back(strack);
    }

    int32_t width, height, channel, rate_den, rate_num;
    if (!input2_bufs->At(i)->Get("width", width)) {
      MBLOG_ERROR
          << "draw_bbox_mot flowunit can not get input 'width' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key width "};
    }
    if (!input2_bufs->At(i)->Get("height", height)) {
      MBLOG_ERROR
          << "draw_bbox_mot flowunit can not get input 'height' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key height "};
    }
    if (!input2_bufs->At(i)->Get("channel", channel)) {
      MBLOG_ERROR
          << "draw_bbox_mot flowunit can not get input 'channel' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key channel "};
    }
    if (!input2_bufs->At(i)->Get("rate_den", rate_den)) {
      MBLOG_ERROR
          << "draw_bbox_mot flowunit can not get input 'rate_den' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key rate_den "};
    }
    if (!input2_bufs->At(i)->Get("rate_num", rate_num)) {
      MBLOG_ERROR
          << "draw_bbox_mot flowunit can not get input 'rate_num' from meta";
      return {modelbox::STATUS_NOTSUPPORT, "meta don't have key rate_num "};
    }

    cv::Mat image(height, width, CV_8UC3);
    auto img_ret = memcpy_s(image.data, image.total() * image.elemSize(),
                            input2_bufs->ConstBufferData(i),
                            input2_bufs->At(i)->GetBytes());
    if (EOK != img_ret) {
      MBLOG_ERROR << "Cpu draw_bbox_mot failed, img_ret " << img_ret;
      return modelbox::STATUS_FAULT;
    }

    for (auto &strack : output_stracks) {
      int track_id = strack->mTrackId;
      std::string track_idstr = std::to_string(track_id);
      cv::Scalar color =
          cv::Scalar((prime_one_ * track_id * ratio_) % gray_range_,
                     (prime_two_ * track_id * ratio_) % gray_range_,
                     (prime_three_ * track_id * ratio_) % gray_range_);
      cv::rectangle(image,
                    cv::Point(strack->mPersonbox.x, strack->mPersonbox.y),
                    cv::Point(strack->mPersonbox.x + strack->mPersonbox.w,
                              strack->mPersonbox.y + strack->mPersonbox.h),
                    color, thickness_, linetype_, shift_);
      cv::putText(image, track_idstr,
                  cv::Point(strack->mPersonbox.x + strack->mPersonbox.w / 2,
                            strack->mPersonbox.y + strack->mPersonbox.h / 2),
                  cv::FONT_HERSHEY_PLAIN, fontScale_, color, thickness_);
    }

    auto output_data = output_bufs->MutableBufferData(i);
    auto output_ret = memcpy_s(output_data, output_bufs->At(i)->GetBytes(),
                               image.data, image.total() * image.elemSize());
    if (EOK != output_ret) {
      MBLOG_ERROR << "Cpu draw_bbox_mot failed, output_ret " << output_ret;
      return modelbox::STATUS_FAULT;
    }
    output_bufs->At(i)->Set("width", width);
    output_bufs->At(i)->Set("height", height);
    output_bufs->At(i)->Set("channel", channel);
    output_bufs->At(i)->Set("rate_den", rate_den);
    output_bufs->At(i)->Set("rate_num", rate_num);
  }

  MBLOG_DEBUG << "draw Person finish";
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(DrawBBoxMOTFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("In_1", modelbox::DEVICE_TYPE));
  desc.AddFlowUnitInput(modelbox::FlowUnitInput("In_2", modelbox::DEVICE_TYPE));
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
