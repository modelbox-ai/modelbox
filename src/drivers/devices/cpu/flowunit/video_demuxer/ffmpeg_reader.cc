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

#include "ffmpeg_reader.h"

#include "driver_util.h"
#include <modelbox/base/log.h>

#include <regex>

#define GET_FFMPEG_ERR(err_num, var_name)        \
  char var_name[AV_ERROR_MAX_STRING_SIZE] = {0}; \
  av_make_error_string(var_name, AV_ERROR_MAX_STRING_SIZE, err_num);

static int CheckTimeout(void *ctx) {
  if (ctx == nullptr) {
    MBLOG_ERROR << "CheckTimeout: ctx is nullptr!";
    return 1;
  }
  auto *p = (FfmpegReader *)ctx;
  if (p->IsTimeout()) {
    MBLOG_INFO << "CheckTimeout: ffmpeg read timeout !";
    return 1;
  }
  return 0;
}

modelbox::Status FfmpegReader::Open(const std::string &source_url) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
  av_register_all();
#endif
  format_ctx_ = nullptr;

  auto ret = avformat_network_init();
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "avformat_network_init failed, err " << err_str;
    return modelbox::STATUS_FAULT;
  }

  origin_source_url_ = source_url;
  std::regex pattern("://.*?@");
  format_source_url_ = std::regex_replace(origin_source_url_, pattern, "://*@");
  AVDictionary *options = nullptr;
  SetupRtspOption(format_source_url_, &options);
  SetupCommonOption(format_source_url_, &options);
  SetupHttpOption(format_source_url_, &options);

  AVFormatContext *ctx = nullptr;
  ctx = avformat_alloc_context();
  if (ctx == nullptr) {
    av_dict_free(&options);
    return {modelbox::STATUS_FAULT, "ctx is null"};
  }
  ResetStartTime();
  ctx->interrupt_callback.callback = CheckTimeout;
  ctx->interrupt_callback.opaque = this;
  ret = avformat_open_input(&ctx, source_url.c_str(), nullptr, &options);
  av_dict_free(&options);
  if (ret < 0) {
    GET_FFMPEG_ERR(ret, err_str);
    MBLOG_ERROR << "avformat open input[" << format_source_url_
                << "] failed, err " << err_str;
    avformat_close_input(&ctx);
    return modelbox::STATUS_FAULT;
  }

  MBLOG_INFO << "Open source " << format_source_url_ << " success, format "
             << ctx->iformat->long_name << " : " << ctx->iformat->name;
  format_ctx_.reset(ctx,
                    [](AVFormatContext *ctx) { avformat_close_input(&ctx); });
  return modelbox::STATUS_SUCCESS;
}

std::shared_ptr<AVFormatContext> FfmpegReader::GetCtx() { return format_ctx_; }

std::string FfmpegReader::GetSourceURL() { return format_source_url_; }

void FfmpegReader::SetupRtspOption(const std::string &source_url,
                                   AVDictionary **options) {
  const std::string rtsp_prefix = "RTSP:";
  if (source_url.size() < rtsp_prefix.size()) {
    return;
  }

  auto source_url_prefix = source_url.substr(0, rtsp_prefix.size());
  std::transform(source_url_prefix.begin(), source_url_prefix.end(),
                 source_url_prefix.begin(), ::toupper);
  if (source_url_prefix != rtsp_prefix) {
    return;
  }

  MBLOG_INFO << "Source is rtsp stream";
  av_dict_set(options, "rtsp_transport", "tcp", 0);
  av_dict_set(options, "recv_buffer_size", "10240000", 0);
  av_dict_set(options, "stimeout", "2000000", 0);
}

void FfmpegReader::SetupCommonOption(const std::string &source_url,
                                     AVDictionary **options) {
  av_dict_set(options, "reconnect", "1", 0);
  av_dict_set(options, "rw_timeout", "30000000", 0);
  MBLOG_INFO << "Source url:" << driverutil::string_masking(source_url)
             << ", reconnect:true, rw_timeout:30s";
}

void FfmpegReader::SetupHttpOption(const std::string &source_url,
                                   AVDictionary **options) {
  const std::string http_prefix = "http:";
  if (source_url.size() < http_prefix.size()) {
    return;
  }

  auto source_url_prefix = source_url.substr(0, http_prefix.size());
  if (source_url_prefix != http_prefix) {
    return;
  }

  MBLOG_INFO << "Source is http file";
  av_dict_set(options, "multiple_requests", "1", 0);
  av_dict_set(options, "rw_timeout", "1000000", 0);
  av_log_set_level(AV_LOG_ERROR);
}

void FfmpegReader::ResetStartTime() {
  start_time_ = std::chrono::steady_clock::now();
}

bool FfmpegReader::IsTimeout() {
  return (std::chrono::steady_clock::now() - start_time_ >=
          FFMPEG_READER_TIMEOUT_INTERVAL);
}
