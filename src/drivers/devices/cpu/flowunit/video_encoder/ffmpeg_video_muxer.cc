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


#include "ffmpeg_video_muxer.h"
#include <modelbox/base/log.h>
#include "video_decode_common.h"

using namespace modelbox;

Status FfmpegVideoMuxer::Init(const std::shared_ptr<AVCodecContext> &codec_ctx,
                              std::shared_ptr<FfmpegWriter> writer) {
  destination_url_ = writer->GetDestinationURL();
  format_ctx_ = writer->GetCtx();
  auto ret = SetupStreamParam(codec_ctx);
  if (ret != STATUS_SUCCESS) {
    return ret;
  }

  return STATUS_SUCCESS;
}

modelbox::Status FfmpegVideoMuxer::SetupStreamParam(
    const std::shared_ptr<AVCodecContext> &codec_ctx) {
  stream_ = avformat_new_stream(format_ctx_.get(), codec_ctx->codec);
  if (stream_ == nullptr) {
    MBLOG_ERROR << "Create video stream failed";
    return STATUS_FAULT;
  }

  stream_->time_base = codec_ctx->time_base;
  auto ret =
      avcodec_parameters_from_context(stream_->codecpar, codec_ctx.get());
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, ffmpeg_err);
    MBLOG_ERROR << "avcodec_parameters_from_context err " << ffmpeg_err;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

Status FfmpegVideoMuxer::Mux(const AVRational &time_base,
                             const std::shared_ptr<AVPacket> &av_packet) {
  av_packet_rescale_ts(av_packet.get(), time_base, stream_->time_base);
  av_packet->stream_index = stream_->index;
  if (!is_header_wrote_) {
    #ifndef ANDROID
    auto ret = avformat_write_header(format_ctx_.get(), nullptr);
    #else
    AVDictionary *options = nullptr;
    av_dict_set(&options, "rtsp_transport", "tcp", 0);
    av_dict_set(&options, "recv_buffer_size", "10240000", 0);
    av_dict_set(&options, "stimeout", "2000000", 0);
    auto ret = avformat_write_header(format_ctx_.get(), &options);
    #endif

    if (ret < 0) {
      GET_FFMPEG_ERR(ret, ffmpeg_err);
      MBLOG_ERROR << "avformat_write_header failed, ret " << ffmpeg_err;
      return STATUS_FAULT;
    }

    is_header_wrote_ = true;
  }

  auto ret = av_interleaved_write_frame(format_ctx_.get(), av_packet.get());
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, ffmpeg_err);
    MBLOG_ERROR << "av_write_frame failed, ret " << ffmpeg_err;
    return STATUS_FAULT;
  }

  return STATUS_SUCCESS;
}

FfmpegVideoMuxer::~FfmpegVideoMuxer() {
  if (is_header_wrote_) {
    auto ret = av_write_trailer(format_ctx_.get());
    if (ret < 0) {
      GET_FFMPEG_ERR(ret, ffmpeg_err);
      MBLOG_ERROR << "av_write_trailer failed, ret " << ffmpeg_err;
    }
  }
}