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

#include "ffmpeg_video_encoder.h"

#include <modelbox/base/log.h>

#include "video_decode_common.h"

modelbox::Status FfmpegVideoEncoder::Init(int32_t width, int32_t height,
                                          const AVRational &frame_rate,
                                          uint64_t bit_rate,
                                          const std::string &encoder_name) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
  av_register_all();
#endif
  auto *codec = avcodec_find_encoder_by_name(encoder_name.c_str());
  if (codec == nullptr) {
    MBLOG_ERROR << "Find encoder failed, encoder name:" << encoder_name;
    return modelbox::STATUS_FAULT;
  }

  auto *codec_ctx = avcodec_alloc_context3(codec);
  if (codec_ctx == nullptr) {
    MBLOG_ERROR << "Alloc codec ctx failed, encoder name:" << encoder_name;
    return modelbox::STATUS_FAULT;
  }

  codec_ctx_.reset(codec_ctx,
                   [](AVCodecContext *ctx) { avcodec_free_context(&ctx); });
  AVDictionary *param = nullptr;
  SetupCodecParam(width, height, frame_rate, bit_rate, param, codec_ctx_);
  auto ffmpeg_ret = avcodec_open2(codec_ctx_.get(), codec, &param);
  av_dict_free(&param);
  if (ffmpeg_ret < 0) {
    GET_FFMPEG_ERR(ffmpeg_ret, ffmpeg_err);
    MBLOG_ERROR << "avcodec_open2 failed, ret " << ffmpeg_err;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

void FfmpegVideoEncoder::SetupCodecParam(
    int32_t width, int32_t height, const AVRational &frame_rate,
    uint64_t bit_rate, AVDictionary *&param,
    std::shared_ptr<AVCodecContext> &codec_ctx) {
  av_dict_set(&param, "preset", "fast", 0);
  codec_ctx->framerate = frame_rate;
  codec_ctx_->bit_rate = bit_rate;
  codec_ctx->time_base = av_inv_q(frame_rate);
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  codec_ctx->max_b_frames = 0;
}

modelbox::Status FfmpegVideoEncoder::Encode(
    const std::shared_ptr<AVFrame> &av_frame,
    std::vector<std::shared_ptr<AVPacket>> &av_packet_list) {
  auto ret = avcodec_send_frame(codec_ctx_.get(), av_frame.get());
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, ffmpeg_err);
    MBLOG_ERROR << "avcodec_send_frame failed, ret " << ffmpeg_err;
    return modelbox::STATUS_FAULT;
  }

  do {
    auto *av_packet_ptr = av_packet_alloc();
    if (av_packet_ptr == nullptr) {
      MBLOG_ERROR << "av packet alloc failed";
      return modelbox::STATUS_FAULT;
    }

    std::shared_ptr<AVPacket> av_packet(
        av_packet_ptr, [](AVPacket *pkt) { av_packet_free(&pkt); });
    ret = avcodec_receive_packet(codec_ctx_.get(), av_packet.get());
    if (ret == AVERROR(EAGAIN)) {
      return modelbox::STATUS_SUCCESS;
    }

    if (ret == AVERROR_EOF) {
      return modelbox::STATUS_NODATA;
    }

    if (ret < 0) {
      GET_FFMPEG_ERR(ret, err_str);
      MBLOG_ERROR << "avcodec_receive_packet failed, err " << err_str;
      return modelbox::STATUS_FAULT;
    }

    av_packet_list.push_back(av_packet);
  } while (ret >= 0);

  return modelbox::STATUS_SUCCESS;
}