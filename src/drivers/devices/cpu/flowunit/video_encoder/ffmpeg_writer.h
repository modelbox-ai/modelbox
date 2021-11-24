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


#ifndef MODELBOX_FLOWUNIT_FFMPEG_WRITER_H_
#define MODELBOX_FLOWUNIT_FFMPEG_WRITER_H_

#include <modelbox/base/status.h>
#include <memory>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
}

class FfmpegWriter {
 public:
  modelbox::Status Open(const std::string &format_name,
                      const std::string &destination_url);

  std::string GetFormatName() { return format_name_; }

  std::string GetDestinationURL() { return destination_url_; }

  std::shared_ptr<AVFormatContext> GetCtx() { return format_ctx_; }

 private:
  std::string format_name_;
  std::string destination_url_;
  std::shared_ptr<AVFormatContext> format_ctx_;
};

#endif  // MODELBOX_FLOWUNIT_FFMPEG_WRITER_H_