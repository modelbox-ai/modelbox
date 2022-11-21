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

#include "video_decoder_flowunit.h"

#include <securec.h>

#include <string>

#include "modelbox/base/log.h"
#include "modelbox/base/status.h"
#include "modelbox/base/utils.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"
#include "rk_video_decoder.h"

constexpr uint32_t MAX_PACKAGE_NUM = 2;

VideoDecoderFlowUnit::VideoDecoderFlowUnit() = default;
VideoDecoderFlowUnit::~VideoDecoderFlowUnit() = default;

modelbox::Status VideoDecoderFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  out_pix_fmt_str_ = opts->GetString("pix_fmt", modelbox::IMG_DEFAULT_FMT);
  MBLOG_INFO << "RKNPU video_decoder with " << out_pix_fmt_str_;

  out_pix_fmt_ = modelbox::GetRGAFormat(out_pix_fmt_str_);
  if (out_pix_fmt_ == RK_FORMAT_UNKNOWN) {
    MBLOG_ERROR << "Not support pix fmt " << out_pix_fmt_str_;
    return modelbox::STATUS_BADCONF;
  }

  queue_size_ = opts->GetUint64("queue_size", DEC_BUF_LIMIT / 2);

  return modelbox::STATUS_OK;
}

modelbox::Status VideoDecoderFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status VideoDecoderFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_INFO << "Video Decode DataPre";
  auto in_meta = ctx->GetInputMeta(VIDEO_PACKET_INPUT);
  auto codec_id =
      std::static_pointer_cast<AVCodecID>(in_meta->GetMeta(CODEC_META));
  if (codec_id == nullptr) {
    MBLOG_ERROR << "Stream codec id is null, init decoder failed";
    return modelbox::STATUS_FAULT;
  }

  auto video_decoder = std::make_shared<RKNPUVideoDecoder>();
  auto ret = video_decoder->Init(*codec_id);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Video decoder init failed";
    return modelbox::STATUS_FAULT;
  }

  auto frame_index = std::make_shared<int64_t>();
  *frame_index = 0;
  ctx->SetPrivate(DECODER_CTX, video_decoder);
  ctx->SetPrivate(FRAME_INDEX_CTX, frame_index);
  return modelbox::STATUS_OK;
};

modelbox::Status VideoDecoderFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> ctx) {
  MBLOG_DEBUG << "rknpu Decode DataPost";
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecoderFlowUnit::WriteData(
    const std::shared_ptr<modelbox::DataContext> &ctx,
    std::shared_ptr<modelbox::Buffer> &pack_buff,
    std::vector<MppFrame> &out_frame) {
  int32_t rate_num = 0;
  int32_t rate_den = 0;
  int64_t duration = 0;
  pack_buff->Get("rate_num", rate_num);
  pack_buff->Get("rate_den", rate_den);
  pack_buff->Get("duration", duration);
  double time_base = 0;
  pack_buff->Get("time_base", time_base);

  auto output_bufs = ctx->Output(FRAME_INFO_OUTPUT);
  auto frame_index =
      std::static_pointer_cast<int64_t>(ctx->GetPrivate(FRAME_INDEX_CTX));

  for (auto &frame : out_frame) {
    auto pts = (int64_t)(mpp_frame_get_pts(frame) * time_base);

    auto buffer = modelbox::ColorChange(frame, out_pix_fmt_, GetBindDevice());
    // out_frame[i] may be deinit after ColorChange
    if (buffer == nullptr) {
      MBLOG_ERROR << "failed to ColorChange";
      continue;
    }

    buffer->Set("index", *frame_index);
    *frame_index = *frame_index + 1;

    buffer->Set("pix_fmt", out_pix_fmt_str_);
    buffer->Set("rate_num", rate_num);
    buffer->Set("rate_den", rate_den);
    buffer->Set("duration", duration);
    buffer->Set("timestamp", pts);
    buffer->Set("eos", false);

    output_bufs->PushBack(buffer);
  }
  return modelbox::STATUS_SUCCESS;
}

// note: it will block for a while (10s) if buffer not release, so must set
// enough thread num
modelbox::Status VideoDecoderFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto video_decoder =
      std::static_pointer_cast<RKNPUVideoDecoder>(ctx->GetPrivate(DECODER_CTX));
  if (video_decoder == nullptr) {
    MBLOG_ERROR << "Video decoder is not init";
    return modelbox::STATUS_FAULT;
  }

  auto video_packet_input = ctx->Input(VIDEO_PACKET_INPUT);
  if (video_packet_input == nullptr) {
    MBLOG_ERROR << "video packet input is null";
    return modelbox::STATUS_FAULT;
  }

  if (video_packet_input->Size() == 0 ||
      video_packet_input->Size() > MAX_PACKAGE_NUM) {
    MBLOG_ERROR << "input size not right: " << video_packet_input->Size()
                << ", set demuxer queue size: 1 ~ " << MAX_PACKAGE_NUM;
    return modelbox::STATUS_FAULT;
  }

  std::lock_guard<std::mutex> lk(rk_dec_mtx_);
  for (size_t i = 0; i < video_packet_input->Size(); ++i) {
    auto packet_buffer = video_packet_input->At(i);
    std::vector<MppFrame> out_frame;

    auto size = packet_buffer->GetBytes();
    if (size <= 1) {
      video_decoder->DecodeFrameBuf(nullptr, 0, out_frame, queue_size_);
    } else {
      video_decoder->DecodeFrameBuf((const uint8_t *)packet_buffer->ConstData(),
                                    size, out_frame, queue_size_);
    }

    if (out_frame.size() > 0) {
      WriteData(ctx, packet_buffer, out_frame);
    }
  }

  return modelbox::STATUS_SUCCESS;
}

MODELBOX_FLOWUNIT(VideoDecoderFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Video");
  desc.AddFlowUnitInput({VIDEO_PACKET_INPUT, "cpu"});
  desc.AddFlowUnitOutput({FRAME_INFO_OUTPUT, modelbox::DEVICE_TYPE});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetInputContiguous(false);
  desc.SetResourceNice(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "pix_fmt", "string", true, modelbox::IMG_DEFAULT_FMT, "the pix format"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(modelbox::DEVICE_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion(MODELBOX_VERSION_STR_MACRO);
}
