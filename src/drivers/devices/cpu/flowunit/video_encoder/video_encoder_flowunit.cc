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

#include "video_encoder_flowunit.h"
#include <securec.h>
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

using namespace modelbox;

VideoEncoderFlowUnit::VideoEncoderFlowUnit(){};
VideoEncoderFlowUnit::~VideoEncoderFlowUnit(){};

const std::set<std::string> g_supported_fmt = {"rtsp", "flv", "mp4"};

modelbox::Status VideoEncoderFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  default_dest_url_ = opts->GetString("default_dest_url", "");

  format_name_ = opts->GetString("format", "rtsp");
  if (format_name_ != "rtsp" && format_name_ != "flv" &&
      format_name_ != "mp4") {
    MBLOG_ERROR << "Bad value [" << format_name_
                << "] for format, must be one of [rtsp|flv|mp4]";
    return STATUS_BADCONF;
  }

  encoder_name_ = opts->GetString("encoder", "mpeg4");
  return STATUS_OK;
}

modelbox::Status VideoEncoderFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status VideoEncoderFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto muxer =
      std::static_pointer_cast<FfmpegVideoMuxer>(ctx->GetPrivate(MUXER_CTX));
  auto encoder = std::static_pointer_cast<FfmpegVideoEncoder>(
      ctx->GetPrivate(ENCODER_CTX));
  auto color_cvt = std::static_pointer_cast<FfmpegColorConverter>(
      ctx->GetPrivate(COLOR_CVT_CTX));
  if (muxer == nullptr || encoder == nullptr || color_cvt == nullptr) {
    MBLOG_ERROR << "Stream not inited";
    return STATUS_FAULT;
  }

  std::vector<std::shared_ptr<AVFrame>> av_frame_list;
  auto ret = ReadFrames(color_cvt, ctx, av_frame_list);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Read input frame failed";
    return STATUS_FAULT;
  }

  std::vector<std::shared_ptr<AVPacket>> av_packet_list;
  ret = EncodeFrame(encoder, av_frame_list, av_packet_list);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Encode frame failed";
    return STATUS_FAULT;
  }

  ret = MuxPacket(muxer, encoder->GetCtx()->time_base, av_packet_list);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Mux packet failed";
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

modelbox::Status VideoEncoderFlowUnit::ReadFrames(
    std::shared_ptr<FfmpegColorConverter> color_cvt,
    std::shared_ptr<modelbox::DataContext> ctx,
    std::vector<std::shared_ptr<AVFrame>> &av_frame_list) {
  auto frame_buffer_list = ctx->Input(FRAME_INFO_INPUT);
  if (frame_buffer_list == nullptr || frame_buffer_list->Size() == 0) {
    MBLOG_ERROR << "Input frame list is empty";
    return STATUS_FAULT;
  }

  auto frame_index_ptr =
      std::static_pointer_cast<int64_t>(ctx->GetPrivate(FRAME_INDEX_CTX));
  for (auto frame_buffer : *frame_buffer_list) {
    std::shared_ptr<AVFrame> av_frame;
    auto ret = ReadFrameFromBuffer(frame_buffer, av_frame);
    av_frame->pts = *frame_index_ptr;
    ++(*frame_index_ptr);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Read frame from buffer failed";
      return ret;
    }

    std::shared_ptr<AVFrame> yuv420p_frame;
    ret = CvtFrameToYUV420P(color_cvt, av_frame, yuv420p_frame);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Convert frame to yuv420p failed";
      return ret;
    }

    av_frame_list.push_back(yuv420p_frame);
  }

  return STATUS_SUCCESS;
}

modelbox::Status VideoEncoderFlowUnit::ReadFrameFromBuffer(
    std::shared_ptr<modelbox::Buffer> &frame_buffer,
    std::shared_ptr<AVFrame> &av_frame) {
  auto frame_ptr = av_frame_alloc();
  if (frame_ptr == nullptr) {
    MBLOG_ERROR << "Alloca frame failed";
    return STATUS_FAULT;
  }

  av_frame.reset(frame_ptr, [](AVFrame *ptr) { av_frame_free(&ptr); });
  frame_buffer->Get("width", av_frame->width);
  frame_buffer->Get("height", av_frame->height);
  std::string pix_fmt;
  frame_buffer->Get("pix_fmt", pix_fmt);
  auto iter = videodecode::g_av_pix_fmt_map.find(pix_fmt);
  if (iter == videodecode::g_av_pix_fmt_map.end()) {
    MBLOG_ERROR << "Encoder not support pix fmt " << pix_fmt;
    return STATUS_NOTSUPPORT;
  }
  av_frame->format = iter->second;
  auto ret =
      av_image_fill_arrays(av_frame->data, av_frame->linesize,
                           (const uint8_t *)frame_buffer->ConstData(),
                           iter->second, av_frame->width, av_frame->height, 1);
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, ffmpeg_err);
    MBLOG_ERROR << "avpicture_fill failed, err " << ffmpeg_err;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

modelbox::Status VideoEncoderFlowUnit::CvtFrameToYUV420P(
    std::shared_ptr<FfmpegColorConverter> color_cvt,
    std::shared_ptr<AVFrame> origin, std::shared_ptr<AVFrame> &yuv420p_frame) {
  auto frame = av_frame_alloc();
  if (frame == nullptr) {
    MBLOG_ERROR << "Alloc frame failed";
    return STATUS_FAULT;
  }

  yuv420p_frame.reset(frame, [](AVFrame *ptr) {
    av_freep(&ptr->data[0]);
    av_frame_free(&ptr);
  });
  yuv420p_frame->width = origin->width;
  yuv420p_frame->height = origin->height;
  yuv420p_frame->format = AVPixelFormat::AV_PIX_FMT_YUV420P;
  yuv420p_frame->pts = origin->pts;
  auto ffmepg_ret = av_image_alloc(yuv420p_frame->data, yuv420p_frame->linesize,
                                   yuv420p_frame->width, yuv420p_frame->height,
                                   AVPixelFormat::AV_PIX_FMT_YUV420P, 1);
  if (ffmepg_ret < 0) {
    GET_FFMPEG_ERR(ffmepg_ret, ffmpeg_err);
    MBLOG_ERROR << "av_image_alloc failed, width " << yuv420p_frame->width
                << ",height " << yuv420p_frame->height << ",err " << ffmpeg_err;
    return STATUS_FAULT;
  }

  auto ret = color_cvt->CvtColor(origin, yuv420p_frame->data[0],
                                 AVPixelFormat::AV_PIX_FMT_YUV420P);
  if (ret != STATUS_SUCCESS) {
    MBLOG_ERROR << "Conver color failed";
    return ret;
  }

  return STATUS_SUCCESS;
}

modelbox::Status VideoEncoderFlowUnit::EncodeFrame(
    const std::shared_ptr<FfmpegVideoEncoder> &encoder,
    const std::vector<std::shared_ptr<AVFrame>> &av_frame_list,
    std::vector<std::shared_ptr<AVPacket>> &av_packet_list) {
  for (auto frame : av_frame_list) {
    auto ret = encoder->Encode(frame, av_packet_list);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Encoder encode frame failed";
      return ret;
    }
  }

  return STATUS_SUCCESS;
}

modelbox::Status VideoEncoderFlowUnit::MuxPacket(
    const std::shared_ptr<FfmpegVideoMuxer> &muxer, const AVRational &time_base,
    std::vector<std::shared_ptr<AVPacket>> &av_packet_list) {
  for (auto packet : av_packet_list) {
    auto ret = muxer->Mux(time_base, packet);
    if (ret != STATUS_SUCCESS) {
      MBLOG_ERROR << "Muxer mux packet failed";
      return ret;
    }
  }

  return STATUS_SUCCESS;
}

modelbox::Status VideoEncoderFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  std::string dest_url;
  auto ret = GetDestUrl(data_ctx, dest_url);
  if (ret != STATUS_SUCCESS) {
    return STATUS_FAULT;
  }

  auto frame_buffer_list = data_ctx->Input(FRAME_INFO_INPUT);
  if (frame_buffer_list == nullptr || frame_buffer_list->Size() == 0) {
    MBLOG_ERROR << "Input [frame_info] is empty";
    return STATUS_FAULT;
  }

  auto frame_buffer = frame_buffer_list->At(0);
  int32_t width = 0;
  int32_t height = 0;
  int32_t rate_num = 0;
  int32_t rate_den = 0;
  frame_buffer->Get("width", width);
  frame_buffer->Get("height", height);
  frame_buffer->Get("rate_num", rate_num);
  frame_buffer->Get("rate_den", rate_den);

  if (width == 0 || height == 0) {
    MBLOG_ERROR << "buffer meta is invalid";
    return STATUS_INVALID;
  }

  if (rate_num == 0 || rate_den == 0) {
    rate_num = 25;
    rate_den = 1;
  }

  auto encoder = std::make_shared<FfmpegVideoEncoder>();
  ret = encoder->Init(width, height, {rate_num, rate_den}, encoder_name_);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Init encoder failed";
    return STATUS_FAULT;
  }

  auto writer = std::make_shared<FfmpegWriter>();
  ret = writer->Open(format_name_, dest_url);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Open ffmepg writer failed, format " << format_name_
                << ", url " << dest_url;
    return STATUS_FAULT;
  }

  auto muxer = std::make_shared<FfmpegVideoMuxer>();
  ret = muxer->Init(encoder->GetCtx(), writer);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Init muxer failed";
    return STATUS_FAULT;
  }

  auto color_cvt = std::make_shared<FfmpegColorConverter>();

  data_ctx->SetPrivate(MUXER_CTX, muxer);
  data_ctx->SetPrivate(ENCODER_CTX, encoder);
  data_ctx->SetPrivate(COLOR_CVT_CTX, color_cvt);
  auto frame_index_ptr = std::make_shared<int64_t>(0);
  data_ctx->SetPrivate(FRAME_INDEX_CTX, frame_index_ptr);
  MBLOG_INFO << "Video encoder init success"
             << ", width " << width << ", height " << height << ", rate "
             << rate_num << "/" << rate_den << ", format " << format_name_
             << ", destination url " << dest_url << ", encoder "
             << encoder_name_;
  return STATUS_OK;
}

modelbox::Status VideoEncoderFlowUnit::GetDestUrl(
    std::shared_ptr<modelbox::DataContext> &data_ctx, std::string &dest_url) {
  auto stream_meta = data_ctx->GetInputMeta(FRAME_INFO_INPUT);
  if (stream_meta != nullptr) {
    auto dest_url_ptr =
        std::static_pointer_cast<std::string>(stream_meta->GetMeta(DEST_URL));
    if (dest_url_ptr != nullptr) {
      dest_url = *dest_url_ptr;
      return STATUS_SUCCESS;
    }
  }

  MBLOG_WARN << "Input meta [dest_url] should be set in port [in_video_frame] for "
                "each stream, Use default_dest_url in config is only "
                "for debug";
  if (default_dest_url_.empty()) {
    MBLOG_ERROR << "default_dest_url in config is empty, no dest url available";
    return STATUS_BADCONF;
  }

  dest_url = default_dest_url_;
  return STATUS_SUCCESS;
}

modelbox::Status VideoEncoderFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return STATUS_OK;
}

MODELBOX_FLOWUNIT(VideoEncoderFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Video");
  desc.AddFlowUnitInput({FRAME_INFO_INPUT});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "default_dest_url", "string", true, "", "the encoder dest url"));

  std::map<std::string, std::string> fmt_list;

  for (auto &item : g_supported_fmt) {
    fmt_list[item] = item;
  }
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "format", "list", true, "rtsp", "the encoder format", fmt_list));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "encoder", "string", true, "mpeg4", "the encoder method"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
