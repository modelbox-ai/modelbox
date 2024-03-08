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

#include "video_decoder_flowunit.h"

#include <securec.h>

#include "ffmpeg_color_converter.h"
#include "ffmpeg_video_decoder.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"
#include "video_decode_common.h"

VideoDecoderFlowUnit::VideoDecoderFlowUnit() = default;
VideoDecoderFlowUnit::~VideoDecoderFlowUnit() = default;

const std::set<std::string> g_supported_pix_fmt = {"nv12", "rgb", "bgr"};

modelbox::Status VideoDecoderFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  auto fmt = opts->GetString("pix_fmt", "nv12");
  if (fmt == "rgb") {
    out_pix_fmt_ = AVPixelFormat::AV_PIX_FMT_RGB24;
  } else if (fmt == "bgr") {
    out_pix_fmt_ = AVPixelFormat::AV_PIX_FMT_BGR24;
  } else if (fmt == "nv12") {
    out_pix_fmt_ = AVPixelFormat::AV_PIX_FMT_NV12;
  } else {
    MBLOG_ERROR << "Not support pix fmt " << fmt;
    return modelbox::STATUS_BADCONF;
  }

  out_pix_fmt_str_ = fmt;
  return modelbox::STATUS_OK;
}

modelbox::Status VideoDecoderFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status VideoDecoderFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  std::shared_ptr<modelbox::Buffer> flag_buffer = nullptr;
  auto video_decoder = std::static_pointer_cast<FfmpegVideoDecoder>(
      data_ctx->GetPrivate(DECODER_CTX));
  if (video_decoder == nullptr) {
    MBLOG_ERROR << "Video decoder is not init";
    return modelbox::STATUS_FAULT;
  }

  std::vector<std::shared_ptr<AVPacket>> pkt_list;
  auto ret = ReadData(data_ctx, pkt_list, flag_buffer);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Read av_packet input failed";
    return modelbox::STATUS_FAULT;
  }

  if (flag_buffer) {
    if (ReopenDecoder(data_ctx, flag_buffer) != modelbox::STATUS_SUCCESS) {
      MBLOG_ERROR << "Reopen decoder failed";
      return modelbox::STATUS_FAULT;
    }

    video_decoder = std::static_pointer_cast<FfmpegVideoDecoder>(
        data_ctx->GetPrivate(DECODER_CTX));
    if (video_decoder == nullptr) {
      MBLOG_ERROR << "Video decoder is not init";
      return modelbox::STATUS_FAULT;
    }
  }

  std::list<std::shared_ptr<AVFrame>> frame_list;
  modelbox::Status decode_ret = modelbox::STATUS_SUCCESS;
  for (auto &pkt : pkt_list) {
    decode_ret = video_decoder->Decode(pkt, frame_list);
    if (decode_ret == modelbox::STATUS_FAULT) {
      MBLOG_ERROR << "Video decoder failed";
      // TODO: Process decoder fault
    }
  }

  ret = WriteData(data_ctx, frame_list, decode_ret == modelbox::STATUS_NODATA);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Send frame data failed";
    return modelbox::STATUS_FAULT;
  }

  if (decode_ret == modelbox::STATUS_NODATA) {
    MBLOG_INFO << "Video decoder finish";
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecoderFlowUnit::ReadData(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::vector<std::shared_ptr<AVPacket>> &pkt_list,
    std::shared_ptr<modelbox::Buffer> &flag_buffer) {
  bool reset_flag = false;
  auto video_packet_input = data_ctx->Input(VIDEO_PACKET_INPUT);
  if (video_packet_input == nullptr) {
    MBLOG_ERROR << "video packet input is null";
    return modelbox::STATUS_FAULT;
  }

  if (video_packet_input->Size() == 0) {
    MBLOG_ERROR << "video packet input size is 0";
    return modelbox::STATUS_FAULT;
  }

  for (size_t i = 0; i < video_packet_input->Size(); ++i) {
    auto packet_buffer = video_packet_input->At(i);

    if (reset_flag == false) {
      packet_buffer->Get("reset_flag", reset_flag);
      if (reset_flag == true) {
        flag_buffer = packet_buffer;
      }
    }

    std::shared_ptr<AVPacket> pkt;
    auto ret = ReadAVPacket(packet_buffer, pkt);
    if (ret != modelbox::STATUS_SUCCESS) {
      return modelbox::STATUS_FAULT;
    }

    pkt_list.push_back(pkt);
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecoderFlowUnit::ReadAVPacket(
    const std::shared_ptr<modelbox::Buffer> &packet_buffer,
    std::shared_ptr<AVPacket> &pkt) {
  auto size = packet_buffer->GetBytes();
  if (size == 1) {
    pkt = std::make_shared<AVPacket>();
    pkt->data = nullptr;
    pkt->size = 0;
    return modelbox::STATUS_SUCCESS;
  }

  auto *data = const_cast<void *>(packet_buffer->ConstData());
  if (data == nullptr) {
    MBLOG_ERROR << "video_packet data is nullptr";
    return modelbox::STATUS_FAULT;
  }

  int64_t pts = 0;
  int64_t dts = 0;
  packet_buffer->Get("pts", pts);
  packet_buffer->Get("dts", dts);
  return BuildAVPacket(pkt, size, (uint8_t *)data, pts, dts);
}

modelbox::Status VideoDecoderFlowUnit::BuildAVPacket(
    std::shared_ptr<AVPacket> &pkt, size_t size, uint8_t *data, int64_t pts,
    int64_t dts) {
  auto *pkt_ptr = av_packet_alloc();
  if (pkt_ptr == nullptr) {
    MBLOG_ERROR << "av_packet_alloc failed";
    return modelbox::STATUS_FAULT;
  }

  pkt.reset(pkt_ptr, [](AVPacket *pkt) {
    pkt->data = nullptr;
    av_packet_free(&pkt);
  });
  pkt->data = data;
  pkt->size = size;
  pkt->pts = pts;
  pkt->dts = dts;
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecoderFlowUnit::WriteData(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::list<std::shared_ptr<AVFrame>> &frame_list, bool eos) {
  auto last_frame =
      std::static_pointer_cast<AVFrame>(data_ctx->GetPrivate(LAST_FRAME));
  data_ctx->SetPrivate(LAST_FRAME, nullptr);
  auto color_cvt = std::static_pointer_cast<FfmpegColorConverter>(
      data_ctx->GetPrivate(CVT_CTX));
  auto frame_buff_list = data_ctx->Output(FRAME_INFO_OUTPUT);
  if (!eos && !frame_list.empty()) {
    // try save last frame in data_ctx, when demuxe end, we could set last
    // frame eos to 'true'
    data_ctx->SetPrivate(LAST_FRAME, frame_list.back());
    frame_list.pop_back();
  }

  if (last_frame != nullptr) {
    frame_list.push_front(last_frame);
  }

  if (frame_list.size() == 0) {
    return modelbox::STATUS_SUCCESS;
  }

  auto frame_index =
      std::static_pointer_cast<int64_t>(data_ctx->GetPrivate(FRAME_INDEX_CTX));
  auto pack_buff_list = data_ctx->Input(VIDEO_PACKET_INPUT);
  auto pack_buff = pack_buff_list->At(0);
  int32_t rate_num = 0;
  int32_t rate_den = 0;
  int32_t rotate_angle = 0;
  int64_t duration = 0;
  pack_buff->Get("rate_num", rate_num);
  pack_buff->Get("rate_den", rate_den);
  pack_buff->Get("rotate_angle", rotate_angle);
  pack_buff->Get("duration", duration);
  double time_base = 0;
  pack_buff->Get("time_base", time_base);
  std::vector<size_t> shape;
  size_t buffer_size;
  for (auto &frame : frame_list) {
    auto ret = videodecode::GetBufferSize(frame->width, frame->height,
                                          out_pix_fmt_str_, buffer_size);
    if (ret != modelbox::STATUS_SUCCESS) {
      return ret;
    }

    shape.push_back(buffer_size);
  }

  frame_buff_list->Build(shape);
  size_t i = 0;
  auto meta = data_ctx->GetInputMeta(VIDEO_PACKET_INPUT);
  auto source_url =
      std::static_pointer_cast<std::string>(meta->GetMeta(SOURCE_URL_META));
  for (auto &frame_ptr : frame_list) {
    videodecode::UpdateStatsInfo(data_ctx, frame_ptr->width, frame_ptr->height);
    auto frame_buff = frame_buff_list->At(i);
    ++i;
    auto ret = color_cvt->CvtColor(
        frame_ptr, (uint8_t *)(frame_buff->MutableData()), out_pix_fmt_);
    if (ret != modelbox::STATUS_SUCCESS) {
      return ret;
    }

    frame_buff->Set("index", *frame_index);
    *frame_index = *frame_index + 1;
    frame_buff->Set("width", frame_ptr->width);
    frame_buff->Set("height", frame_ptr->height);
    frame_buff->Set("height_stride", frame_ptr->height);
    frame_buff->Set("rate_num", rate_num);
    frame_buff->Set("rate_den", rate_den);
    frame_buff->Set("rotate_angle", rotate_angle);
    frame_buff->Set("duration", duration);
    frame_buff->Set("eos", false);
    frame_buff->Set("pix_fmt", out_pix_fmt_str_);
    frame_buff->Set("url", *source_url);
    auto width_stride = frame_ptr->width;
    if (out_pix_fmt_str_ == "rgb" || out_pix_fmt_str_ == "bgr") {
      width_stride *= 3;
      int32_t channel = 3;
      frame_buff->Set("channel", channel);
      frame_buff->Set(
          "shape", std::vector<size_t>({static_cast<size_t>(frame_ptr->height),
                                        static_cast<size_t>(frame_ptr->width),
                                        static_cast<size_t>(channel)}));
      frame_buff->Set("layout", std::string("hwc"));
    }
    frame_buff->Set("width_stride", width_stride);

    frame_buff->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
    frame_buff->Set("timestamp", (int64_t)(frame_ptr->pts * time_base));
    if (eos && frame_ptr == frame_list.back()) {
      frame_buff->Set("eos", true);
    }
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecoderFlowUnit::ReopenDecoder(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::shared_ptr<modelbox::Buffer> &flag_buffer) {
  auto old_source_url = std::static_pointer_cast<std::string>(
      data_ctx->GetPrivate(SOURCE_URL_META));
  auto old_codec_id =
      std::static_pointer_cast<AVCodecID>(data_ctx->GetPrivate(CODEC_ID_META));

  if (old_source_url == nullptr || old_codec_id == nullptr) {
    MBLOG_ERROR << "Reopen decoder failed, source url or codec id is null";
    return modelbox::STATUS_FAULT;
  }

  std::string source_url;
  AVCodecID codec_id;
  if (flag_buffer->Get(SOURCE_URL_META, source_url) == false) {
    return modelbox::STATUS_SUCCESS;
  }

  if (flag_buffer->Get(CODEC_ID_META, codec_id) == false) {
    return modelbox::STATUS_SUCCESS;
  }

  if (source_url == *old_source_url && codec_id == *old_codec_id) {
    return modelbox::STATUS_SUCCESS;
  }

  MBLOG_WARN << "Reopen decoder, source url or codec id changed";
  auto ret = CloseDecoder(data_ctx);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Close decoder failed";
    return modelbox::STATUS_FAULT;
  }

  return NewDecoder(data_ctx, source_url, codec_id);
}

modelbox::Status VideoDecoderFlowUnit::CloseDecoder(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  data_ctx->SetPrivate(DECODER_CTX, nullptr);
  data_ctx->SetPrivate(CVT_CTX, nullptr);
  data_ctx->SetPrivate(FRAME_INDEX_CTX, nullptr);
  data_ctx->SetPrivate(SOURCE_URL_META, nullptr);
  data_ctx->SetPrivate(CODEC_ID_META, nullptr);
  data_ctx->SetOutputMeta(FRAME_INFO_OUTPUT, nullptr);
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecoderFlowUnit::NewDecoder(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::string &source_url, AVCodecID codec_id) {
  auto video_decoder = std::make_shared<FfmpegVideoDecoder>();
  auto ret = video_decoder->Init(codec_id);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Video decoder init failed";
    return modelbox::STATUS_FAULT;
  }

  auto color_cvt = std::make_shared<FfmpegColorConverter>();
  auto frame_index = std::make_shared<int64_t>();
  *frame_index = 0;
  data_ctx->SetPrivate(DECODER_CTX, video_decoder);
  data_ctx->SetPrivate(CVT_CTX, color_cvt);
  data_ctx->SetPrivate(FRAME_INDEX_CTX, frame_index);
  data_ctx->SetPrivate(SOURCE_URL_META,
                       std::make_shared<std::string>(source_url));
  data_ctx->SetPrivate(CODEC_ID_META, std::make_shared<AVCodecID>(codec_id));
  auto meta = std::make_shared<modelbox::DataMeta>();
  meta->SetMeta(SOURCE_URL_META, std::make_shared<std::string>(source_url));
  data_ctx->SetOutputMeta(FRAME_INFO_OUTPUT, meta);
  MBLOG_INFO << "Video decoder init success";
  MBLOG_INFO << "Video decoder output pix fmt " << out_pix_fmt_str_;
  return modelbox::STATUS_OK;
}

modelbox::Status VideoDecoderFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto in_meta = data_ctx->GetInputMeta(VIDEO_PACKET_INPUT);
  auto codec_id =
      std::static_pointer_cast<AVCodecID>(in_meta->GetMeta(CODEC_META));
  if (codec_id == nullptr) {
    MBLOG_ERROR << "Stream codec id is null, init decoder failed";
    return modelbox::STATUS_FAULT;
  }

  auto source_url =
      std::static_pointer_cast<std::string>(in_meta->GetMeta(SOURCE_URL_META));
  if (source_url == nullptr) {
    MBLOG_ERROR << "Stream source url is null, init decoder failed";
    return modelbox::STATUS_FAULT;
  }

  return NewDecoder(data_ctx, *source_url, *codec_id);
}

modelbox::Status VideoDecoderFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return CloseDecoder(data_ctx);
}

MODELBOX_FLOWUNIT(VideoDecoderFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Video");
  desc.AddFlowUnitInput({VIDEO_PACKET_INPUT});
  desc.AddFlowUnitOutput({FRAME_INFO_OUTPUT});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.SetInputContiguous(false);
  std::map<std::string, std::string> pix_fmt_list;

  for (const auto &item : g_supported_pix_fmt) {
    pix_fmt_list[item] = item;
  }
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "pix_fmt", "list", true, "0", "the decoder pixel format", pix_fmt_list));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
