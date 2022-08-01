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

#include "ffmpeg_color_converter.h"

#include <modelbox/base/log.h>
#include <video_decode_common.h>

modelbox::Status FfmpegColorConverter::CvtColor(
    const std::shared_ptr<AVFrame> &src_frame, uint8_t *out_frame_data,
    AVPixelFormat out_pix_fmt) {
  if (!SupportCvtPixFmt(out_pix_fmt)) {
    return modelbox::STATUS_INVALID;
  }

  auto &width = src_frame->width;
  auto &height = src_frame->height;
  if (width_ != width || height != height_) {
    width_ = width;
    height_ = height;
    auto ret = InitSwsCtx(width, height, (AVPixelFormat)src_frame->format,
                          out_pix_fmt);
    if (ret != modelbox::STATUS_SUCCESS) {
      return ret;
    }
  }

  int32_t linesize[4];
  GetLineSize(out_pix_fmt, width, linesize, 4);
  uint8_t *data[4] = {nullptr};
  data[0] = out_frame_data;
  if (out_pix_fmt == AVPixelFormat::AV_PIX_FMT_NV12) {
    data[1] = out_frame_data + width * height;  // For UV plane
  } else if (out_pix_fmt == AVPixelFormat::AV_PIX_FMT_YUV420P) {
    data[1] = out_frame_data + width * height;  // For U plane
    data[2] = data[1] + width * height / 4;     // For V plane
  }

  auto ffmpeg_ret = sws_scale(sws_ctx_.get(), src_frame->data,
                              src_frame->linesize, 0, height, data, linesize);
  if (ffmpeg_ret < 0) {
    GET_FFMPEG_ERR(ffmpeg_ret, ffmpeg_err_str);
    MBLOG_ERROR << "sws_scale failed, detail:" << ffmpeg_err_str;
    return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}

bool FfmpegColorConverter::SupportCvtPixFmt(AVPixelFormat pix_fmt) {
  if (pix_fmt == AVPixelFormat::AV_PIX_FMT_BGR24 ||
      pix_fmt == AVPixelFormat::AV_PIX_FMT_RGB24 ||
      pix_fmt == AVPixelFormat::AV_PIX_FMT_NV12 ||
      pix_fmt == AVPixelFormat::AV_PIX_FMT_YUV420P) {
    return true;
  }

  return false;
}

modelbox::Status FfmpegColorConverter::InitSwsCtx(int32_t width, int32_t height,
                                                  AVPixelFormat src_pix_fmt,
                                                  AVPixelFormat dest_pix_fmt) {
  auto *sws_ctx = sws_getContext(width, height, src_pix_fmt, width, height,
                                 dest_pix_fmt, 0, nullptr, nullptr, nullptr);
  if (sws_ctx == nullptr) {
    auto fmt_name = std::to_string(dest_pix_fmt);
    const auto *name_c = av_get_pix_fmt_name(dest_pix_fmt);
    if (name_c) {
      fmt_name = name_c;
    }
    const char *pix_fmt_name = av_get_pix_fmt_name(src_pix_fmt);
    if (pix_fmt_name == nullptr) {
      pix_fmt_name = "unknown";
    }

    MBLOG_ERROR << "Failed to create sws_ctx for [f:" << fmt_name
                << " w:" << width << " h:" << height << "]->[f:" << pix_fmt_name
                << " w:" << width << " h:" << height << "]";
    return modelbox::STATUS_FAULT;
  }

  sws_ctx_.reset(sws_ctx, [](SwsContext *ctx) { sws_freeContext(ctx); });
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status FfmpegColorConverter::GetLineSize(AVPixelFormat pix_fmt,
                                                   int32_t width,
                                                   int32_t linesize[4],
                                                   int32_t linesize_size) {
  linesize[1] = 0;
  linesize[2] = 0;
  linesize[3] = 0;
  switch (pix_fmt) {
    case AVPixelFormat::AV_PIX_FMT_NV12:
      linesize[0] = width;
      linesize[1] = width;
      break;
    case AVPixelFormat::AV_PIX_FMT_RGB24:
    case AVPixelFormat::AV_PIX_FMT_BGR24:
      linesize[0] = width * 3;
      break;
    case AVPixelFormat::AV_PIX_FMT_YUV420P:
      linesize[0] = width;
      linesize[1] = width / 2;
      linesize[2] = width / 2;
      break;
    default:
      if (av_get_pix_fmt_name(pix_fmt)) {
        MBLOG_ERROR << "Not support pix fmt " << av_get_pix_fmt_name(pix_fmt);
      } else {
        MBLOG_ERROR << "Not support pix fmt " << pix_fmt;
      }
      return modelbox::STATUS_FAULT;
  }

  return modelbox::STATUS_SUCCESS;
}