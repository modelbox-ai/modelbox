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


#ifndef MODELBOX_FLOWUNIT_FFMPEG_READER_H_
#define MODELBOX_FLOWUNIT_FFMPEG_READER_H_

#include <modelbox/base/status.h>

#include <chrono>
#include <memory>
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/log.h>
}
constexpr std::chrono::seconds FFMPEG_READER_TIMEOUT_INTERVAL =
    std::chrono::seconds(60);

class FfmpegReader {
 public:
  modelbox::Status Open(const std::string &source_url);

  std::shared_ptr<AVFormatContext> GetCtx();

  std::string GetSourceURL();

  void ResetStartTime();

  bool IsTimeout();

 private:
  void SetupRtspOption(const std::string &source_url, AVDictionary **options);

  void SetupCommonOption(const std::string &source_url, AVDictionary **options);

  void SetupHttpOption(const std::string &source_url, AVDictionary **options);

  std::string origin_source_url_;
  std::string format_source_url_;
  std::shared_ptr<AVFormatContext> format_ctx_;
  std::chrono::steady_clock::time_point start_time_;
};

#endif