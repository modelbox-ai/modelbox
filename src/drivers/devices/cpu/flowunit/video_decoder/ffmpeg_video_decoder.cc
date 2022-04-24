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


#include "ffmpeg_video_decoder.h"
#include <video_decode_common.h>
#include "modelbox/base/log.h"

using namespace modelbox;

Status FfmpegVideoDecoder::Init(AVCodecID codec_id) {
  codec_id_ = codec_id;
  auto codec_ptr = avcodec_find_decoder(codec_id_);
  if (codec_ptr == nullptr) {
    MBLOG_ERROR << "Find decoder for codec[" << codec_id_ << "] failed";
    return STATUS_FAULT;
  }

  auto av_ctx_ptr = avcodec_alloc_context3(codec_ptr);
  if (av_ctx_ptr == nullptr) {
    MBLOG_ERROR << "avcodec_alloc_context3 return, codec_id "
                << codec_id_;
    return STATUS_FAULT;
  }

  AVDictionary *opts = nullptr;
  av_dict_set(&opts, "refcounted_frames", "1", 0);
  auto ret = avcodec_open2(av_ctx_ptr, codec_ptr, &opts);
  av_dict_free(&opts);
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "avcodec_open2 failed, code_id " << codec_id_ << ", err "
                << err_str;
    avcodec_free_context(&av_ctx_ptr);
    return STATUS_FAULT;
  }

  av_ctx_.reset(av_ctx_ptr,
                [](AVCodecContext *ctx) { avcodec_free_context(&ctx); });

  return STATUS_SUCCESS;
}

Status FfmpegVideoDecoder::Decode(
    const std::shared_ptr<const AVPacket> &av_packet,
    std::list<std::shared_ptr<AVFrame>> &av_frame_list) {
  auto ret = avcodec_send_packet(av_ctx_.get(), av_packet.get());
  if (ret == AVERROR_EOF) {
    return STATUS_NODATA;
  } else if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "avcodec_send_packet failed, err " << err_str;
    return STATUS_FAULT;
  }

  do {
    auto av_frame_ptr = av_frame_alloc();
    if (av_frame_ptr == nullptr) {
      MBLOG_ERROR << "av frame alloc failed";
      return STATUS_FAULT;
    }

    std::shared_ptr<AVFrame> av_frame(
        av_frame_ptr, [](AVFrame *frame) { av_frame_free(&frame); });
    ret = avcodec_receive_frame(av_ctx_.get(), av_frame.get());
    if (ret == AVERROR(EAGAIN)) {
      return STATUS_SUCCESS;
    }

    if (ret == AVERROR_EOF) {
      return STATUS_NODATA;
    }

    if (ret < 0) {
      GET_FFMPEG_ERR(ret, err_str);
      MBLOG_ERROR << "avcodec_receive_frame failed, err " << err_str;
      return STATUS_FAULT;
    }

    av_frame_list.push_back(av_frame);
  } while (ret >= 0);

  return STATUS_SUCCESS;
}