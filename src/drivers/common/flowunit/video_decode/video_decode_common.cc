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


#include "video_decode_common.h"

#include <modelbox/base/log.h>

#include <functional>
#include <map>

namespace videodecode {

size_t NV12BufferSize(int32_t width, int32_t height) {
  return width * height * 3 / 2;
}

size_t RGBBufferSize(int32_t width, int32_t height) {
  return width * height * 3;
}

std::map<std::string, std::function<size_t(int32_t width, int32_t height)>>
    g_pix_fmt_to_buffer_size = {{"nv12", NV12BufferSize},
                                {"rgb", RGBBufferSize},
                                {"bgr", RGBBufferSize}};

const std::set<std::string> g_supported_pix_fmt = {"nv12", "rgb", "bgr"};
const std::map<std::string, AVPixelFormat> g_av_pix_fmt_map = {
    {"nv12", AVPixelFormat::AV_PIX_FMT_NV12},
    {"rgb", AVPixelFormat::AV_PIX_FMT_RGB24},
    {"bgr", AVPixelFormat::AV_PIX_FMT_BGR24}};

modelbox::Status GetBufferSize(int32_t width, int32_t height,
                             const std::string &pix_fmt, size_t &size) {
  auto iter = g_pix_fmt_to_buffer_size.find(pix_fmt);
  if (iter == g_pix_fmt_to_buffer_size.end()) {
    MBLOG_ERROR << "Not support pix fmt " << pix_fmt;
    return modelbox::STATUS_NOTSUPPORT;
  }

  size = iter->second(width, height);
  return modelbox::STATUS_SUCCESS;
}

void UpdateStatsInfo(std::shared_ptr<modelbox::DataContext> &ctx, int32_t width,
                     int32_t height) {
  auto stats = ctx->GetStatistics();
  stats->AddItem("frame_width", width, true);
  stats->AddItem("frame_height", height, true);
  uint64_t one_frame = 1;
  stats->IncreaseValue("frame_count", one_frame);
}

}  // namespace videodecode