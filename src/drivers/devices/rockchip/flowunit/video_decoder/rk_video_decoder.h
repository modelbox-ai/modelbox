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

#ifndef MODELBOX_RK_VIDEO_DECODER_H_
#define MODELBOX_RK_VIDEO_DECODER_H_

#include <libavformat/avformat.h>
#include <modelbox/base/status.h>
#include <modelbox/device/rockchip/rockchip_api.h>

#include <vector>

#include "rga.h"
#include "rk_mpi.h"
#include "rk_type.h"

constexpr uint32_t DEC_BUF_LIMIT = 8;

class RKNPUVideoDecoder {
 public:
  RKNPUVideoDecoder() = default;
  virtual ~RKNPUVideoDecoder();

  modelbox::Status Init(AVCodecID codec_id);
  modelbox::Status DecodeFrameBuf(const uint8_t *inData, size_t inSize,
                                  std::vector<MppFrame> &out_frame,
                                  size_t max_frames);

 private:
  modelbox::Status InitDecoder(MppCodingType codec_type);
  void SetPacket(MppPacket &packet, uint8_t *inData, size_t inSize);
  modelbox::Status InfoChange(MppFrame &frame);
  modelbox::Status GetDecFrame(MppFrame &frame);
  modelbox::Status SendDecBuf(MppPacket &packet);
  void GetLimitDecFrame(std::vector<MppFrame> &out_frame, size_t max_frames);

 private:
  uint32_t err_number_ = 0;
  bool running_ = false;
  MppCtx codec_ctx_ = nullptr;
  MppApi *rk_api_ = nullptr;
  MppBufferGroup frm_grp_ = nullptr;
  MppEncCfg cfg_ = nullptr;
};

#endif  // MODELBOX_RK_VIDEO_DECODER_H_