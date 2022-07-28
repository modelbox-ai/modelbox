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


#ifndef MODELBOX_FLOWUNIT_VIDEO_DECODE_COMMON_H_
#define MODELBOX_FLOWUNIT_VIDEO_DECODE_COMMON_H_

#include <modelbox/base/status.h>
#include <modelbox/data_context.h>

#include <map>
#include <set>
#include <string>

#include "ffmpeg_color_converter.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
}

namespace videodecode {

extern const std::set<std::string> g_supported_pix_fmt;
extern const std::map<std::string, AVPixelFormat> g_av_pix_fmt_map;

#define GET_FFMPEG_ERR(err_num, var_name)        \
  char var_name[AV_ERROR_MAX_STRING_SIZE] = {0}; \
  av_make_error_string(var_name, AV_ERROR_MAX_STRING_SIZE, err_num);

modelbox::Status GetBufferSize(int32_t width, int32_t height,
                             const std::string &pix_fmt, size_t &size);

void UpdateStatsInfo(std::shared_ptr<modelbox::DataContext> &data_ctx,
                     int32_t width, int32_t height);
}  // namespace videodecode

#endif  // MODELBOX_FLOWUNIT_VIDEO_DECODE_COMMON_H_