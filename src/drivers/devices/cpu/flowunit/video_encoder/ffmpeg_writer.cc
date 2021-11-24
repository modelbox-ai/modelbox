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


#include "ffmpeg_writer.h"
#include <modelbox/base/log.h>
#include "video_decode_common.h"

extern "C" {
#include <libavutil/opt.h>
}

using namespace modelbox;

Status FfmpegWriter::Open(const std::string &format_name,
                          const std::string &destination_url) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
  av_register_all();
#endif
  auto ret = avformat_network_init();
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, ffmpeg_err);
    MBLOG_ERROR << "avformat_network_init, err " << ffmpeg_err;
    return STATUS_FAULT;
  }

  format_name_ = format_name;
  destination_url_ = destination_url;

  AVFormatContext *format_ctx = nullptr;
  ret = avformat_alloc_output_context2(
      &format_ctx, nullptr, format_name.c_str(), destination_url.c_str());
  if (ret < 0 || format_ctx == nullptr) {
    GET_FFMPEG_ERR(ret, ffmpeg_err);
    MBLOG_ERROR << "avformat_alloc_output_context2 failed, format "
                << format_name << ", dest_url " << destination_url << ", ret "
                << ffmpeg_err;
    return STATUS_FAULT;
  }

  format_ctx_.reset(format_ctx,
                    [](AVFormatContext *ctx) { avformat_free_context(ctx); });
  if (format_name_ != "rtsp") {
    ret = avio_open2(&format_ctx_->pb, destination_url.c_str(), AVIO_FLAG_WRITE,
                     nullptr, nullptr);
    if (ret < 0) {
      GET_FFMPEG_ERR(ret, ffmpeg_err);
      MBLOG_ERROR << "avio_open2 failed, url " << destination_url << ", format "
                  << format_name << ", ret " << ffmpeg_err;
      return STATUS_FAULT;
    }
  }

  MBLOG_INFO << "Open url " << destination_url << ", format " << format_name
             << " success";
  return STATUS_SUCCESS;
}