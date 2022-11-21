/*
 * Copyright 2022 The Modelbox Project Authors. All Rights Reserved.
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

#ifndef MODELBOX_FLOWUNIT_LOCAL_CAMERA_ROCKCHIP_H_
#define MODELBOX_FLOWUNIT_LOCAL_CAMERA_ROCKCHIP_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/device/rockchip/device_rockchip.h>
#include <modelbox/device/rockchip/rockchip_api.h>

#include <algorithm>
#include <string>

#include "rga.h"

constexpr const char *FLOWUNIT_NAME = "local_camera";
constexpr const char *FLOWUNIT_DESC = "A rockchip local camera flowunit";
constexpr const char *LOCAL_CAMERA_INPUT = "in_camera_packet";
constexpr const char *FRAME_INFO_OUTPUT = "out_camera_frame";
constexpr const char *LOCAL_CAMERA_CTX = "local_camera_ctx";
constexpr const char *FRAME_INDEX_CTX = "frame_index_ctx";
constexpr const char *RETRY_COUNT_CTX = "retry_count_ctx";
constexpr const char *SOURCE_URL = "source_url";

class RockChipLocalCameraFlowUnit : public modelbox::FlowUnit {
 public:
  RockChipLocalCameraFlowUnit();
  ~RockChipLocalCameraFlowUnit() override;

  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  modelbox::Status BuildOutput(
      const std::shared_ptr<modelbox::DataContext> &data_ctx,
      std::shared_ptr<modelbox::Buffer> &img_buf, MppFrame &frame,
      std::shared_ptr<int64_t> &frame_index);
  MppFrame ProcessJpg(const uint8_t *buf, size_t size, size_t w, size_t h,
                      std::shared_ptr<modelbox::Buffer> &img_buf);
  MppFrame ProcessNV12(const uint8_t *buf, size_t size, size_t w, size_t h,
                       std::shared_ptr<modelbox::Buffer> &img_buf);
  MppFrame ProcessRGB24(const uint8_t *buf, size_t size, size_t w, size_t h,
                        std::shared_ptr<modelbox::Buffer> &img_buf);
  MppFrame ProcessYVY2(const uint8_t *buf, size_t size, size_t w, size_t h,
                       std::shared_ptr<modelbox::Buffer> &img_buf);
  MppFrame SetMppFrameInfo(size_t w, size_t h, MppFrameFormat fmt,
                           MppBuffer mpp_buf);

  MppFrame SetMppFrameInfo(size_t w, size_t h, MppFrameFormat fmt,
                           MppBuffer mpp_buf);

  uint32_t camWidth_{0};
  uint32_t camHeight_{0};
  uint32_t camera_id_{0};
  uint32_t fps_{30};
  bool mirror_{true};
  std::string camera_bus_info_;
  modelbox::MppJpegDecode jpeg_dec_;
  std::string out_pix_fmt_str_;
  RgaSURF_FORMAT out_pix_fmt_{RK_FORMAT_YCbCr_420_SP};
  std::mutex jpgdec_mtx_;
};

#endif  // MODELBOX_FLOWUNIT_LOCAL_CAMERA_ROCKCHIP_H_
