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

#ifndef MODELBOX_ROCKCHIP_API_H_
#define MODELBOX_ROCKCHIP_API_H_

#include <modelbox/base/status.h>
#include <modelbox/buffer.h>

#include "im2d.h"
#include "rga.h"
#include "rk_mpi.h"
#include "rk_type.h"

#define MPPFRAMETORGA(frame, fmt)                               \
  wrapbuffer_fd(mpp_buffer_get_fd(mpp_frame_get_buffer(frame)), \
                (int)mpp_frame_get_width(frame),                \
                (int)mpp_frame_get_height(frame), fmt,          \
                (int)mpp_frame_get_hor_stride(frame),           \
                (int)mpp_frame_get_ver_stride(frame));

#define MPP_ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))
#define MPP_ALIGN_MPP_WH 16
#define MPP_ALIGN_WIDTH 16
#define MPP_ALIGN_HEIGHT 2
#define RK_POLL_TIMEOUT 500

namespace modelbox {

constexpr const char *IMG_DEFAULT_FMT = "bgr";

std::shared_ptr<modelbox::Buffer> CreateEmptyMppImg(
    int w, int h, RgaSURF_FORMAT fmt, const std::shared_ptr<Device> &dev,
    rga_buffer_t &rga_buf);

RgaSURF_FORMAT GetRGAFormat(const std::string &fmt_str);
RgaSURF_FORMAT GetRGAFormat(const MppFrameFormat &fmt_mpp);

Status CopyRGBMemory(uint8_t *psrc, uint8_t *pdst, int w, int h, int ws,
                     int hs);
Status CopyNVMemory(uint8_t *psrc, uint8_t *pdst, int w, int h, int ws, int hs);

Status GetRGAFromImgBuffer(std::shared_ptr<Buffer> &in_img, RgaSURF_FORMAT fmt,
                           rga_buffer_t &rgb_buf);

std::shared_ptr<modelbox::Buffer> ColorChange(
    MppFrame &frame, RgaSURF_FORMAT fmt, const std::shared_ptr<Device> &device);
std::shared_ptr<modelbox::Buffer> MirrorImg(
    const std::shared_ptr<modelbox::Buffer> &in_buf, RgaSURF_FORMAT fmt);

class MppJpegDecode {
 public:
  MppJpegDecode() = default;
  virtual ~MppJpegDecode();
  Status Init();
  MppFrame Decode(void *in_buf, int buf_len, int &w, int &h);
  MppFrame Decode(MppBuffer &in_buf, int &w, int &h);

 private:
  void GetJpegWH(int &nW, int &nH, const unsigned char *buf, int bufLen);
  Status DecPkt(MppPacket &packet, int w = 0, int h = 0);
  Status ShutDown();
  MppPacket SendBuf(MppBuffer &in_buf, int &w, int &h);
  MppPacket SendBuf(void *in_buf, int buf_len, int &w, int &h);
  Status ReceiveFrame(MppFrame &out_frame);

  MppCtx codec_ctx_{nullptr};
  MppApi *rk_api_{nullptr};
  MppBufferGroup frm_grp_{nullptr};
  std::mutex jpeg_mtx_;
};

class InferenceRKNPUParams {
 public:
  InferenceRKNPUParams() = default;
  virtual ~InferenceRKNPUParams() = default;

  std::vector<std::string> input_name_list_, output_name_list_;
  std::vector<std::string> input_type_list_, output_type_list_;

  int32_t device_id_{0};
};

class InferenceInputParams {
 public:
  InferenceInputParams() = default;
  virtual ~InferenceInputParams() = default;

  int32_t in_width_ = 0;
  int32_t in_height_ = 0;
  int32_t in_wstride_ = 0;
  int32_t in_hstride_ = 0;
  std::string pix_fmt_;
};

}  // namespace modelbox

#endif  // MODELBOX_ROCKCHIP_API_H_