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

#include "rk_video_decoder.h"

#include <unistd.h>

#include "modelbox/base/log.h"

constexpr uint32_t DEC_RETRYS = 10;
constexpr uint32_t DEC_DELAY_TIMES = 100 * 1000;
constexpr uint32_t RETRY_DELAY_TIMES = 10 * 1000;
constexpr uint32_t ERROR_LOG_TIMES = 100;

RKNPUVideoDecoder::~RKNPUVideoDecoder() {
  running_ = false;

  if (cfg_) {
    mpp_enc_cfg_deinit(cfg_);  // todo: here or Init_Config?
  }
  if (rk_api_ && codec_ctx_) {
    rk_api_->reset(codec_ctx_);
  }
  if (codec_ctx_) {
    mpp_destroy(codec_ctx_);
    codec_ctx_ = nullptr;
  }
  if (frm_grp_) {
    mpp_buffer_group_put(frm_grp_);
    frm_grp_ = nullptr;
  }
}

modelbox::Status RKNPUVideoDecoder::InitDecoder(MppCodingType codec_type) {
  auto ret = mpp_create(&codec_ctx_, &rk_api_);
  if (ret != MPP_OK) {
    auto msg = std::string("failed to run mpp_create: ") + std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  // rockchip codec init
  RK_U32 need_split = 1;
  ret =
      rk_api_->control(codec_ctx_, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);
  if (ret != MPP_OK) {
    auto msg = std::string("failed to set MPP_DEC_SET_PARSER_SPLIT_MODE: ") +
               std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  RK_U32 timeout = 0;
  ret = rk_api_->control(codec_ctx_, MPP_SET_OUTPUT_TIMEOUT, &timeout);
  if (ret != MPP_OK) {
    auto msg = std::string("Failed to set output timeout 0 fail: ") +
               std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  ret = mpp_init(codec_ctx_, MPP_CTX_DEC, codec_type);
  if (ret != MPP_OK) {
    auto msg = std::string("failed to run mpp_init: ") + std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  mpp_dec_cfg_init(&cfg_);
  /*
   * split_parse is to enable mpp internal frame spliter when the
   * input packet_ is not aplited into frames.
   */
  ret = mpp_dec_cfg_set_u32(cfg_, "base:split_parse", need_split);
  if (ret != MPP_OK) {
    auto msg = std::string("failed to set split_parse: ") + std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  ret = rk_api_->control(codec_ctx_, MPP_DEC_SET_CFG, cfg_);
  if (ret != MPP_OK) {
    auto msg = std::string("failed to set cfg: ") + std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  running_ = true;
  return modelbox::STATUS_OK;
}

modelbox::Status RKNPUVideoDecoder::Init(AVCodecID codec_id) {
  const std::map<AVCodecID, MppCodingType> codectype_map = {
      {AV_CODEC_ID_H264, MPP_VIDEO_CodingAVC},
      {AV_CODEC_ID_HEVC, MPP_VIDEO_CodingHEVC}};

  auto iter = codectype_map.find(codec_id);
  if (iter == codectype_map.end()) {
    auto msg =
        std::string("Not support codec type: ") + std::to_string(codec_id);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_NOTSUPPORT, msg};
  }

  err_number = 0;
  return InitDecoder(iter->second);
}

void RKNPUVideoDecoder::SetPacket(MppPacket &packet, uint8_t *inData,
                                  size_t inSize) {
  mpp_packet_set_data(packet, inData);
  mpp_packet_set_size(packet, inSize);
  mpp_packet_set_pos(packet, inData);
  mpp_packet_set_length(packet, inSize);
  if (inSize == 0 || inData == nullptr) {
    mpp_packet_set_eos(packet);
  }
}

modelbox::Status RKNPUVideoDecoder::InfoChange(MppFrame &frame) {
  RK_U32 buf_size = mpp_frame_get_buf_size(frame);

  if (frm_grp_ == nullptr) {
    /* If buffer group is not set create one and limit it */
    auto ret = mpp_buffer_group_get_internal(&frm_grp_, MPP_BUFFER_TYPE_ION);
    if (ret != MPP_OK) {
      auto msg =
          std::string("get mpp buffer group failed: ") + std::to_string(ret);
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }

    /* Set buffer to mpp decoder */
    ret = rk_api_->control(codec_ctx_, MPP_DEC_SET_EXT_BUF_GROUP, frm_grp_);
    if (ret != MPP_OK) {
      auto msg = std::string("set buffer group failed: ") + std::to_string(ret);
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }
  } else {
    /* If old buffer group exist clear it */
    auto ret = mpp_buffer_group_clear(frm_grp_);
    if (ret != MPP_OK) {
      auto msg =
          std::string("clear buffer group failed: ") + std::to_string(ret);
      MBLOG_ERROR << msg;
      return {modelbox::STATUS_FAULT, msg};
    }
  }

  /* Use limit config to limit buffer count to 16~24 with buf_size */
  auto ret =
      mpp_buffer_group_limit_config(frm_grp_, buf_size, DEC_BUF_LIMIT * 2);
  if (ret != MPP_OK) {
    auto msg = std::string("limit buffer group failed: ") + std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  /*
   * All buffer group config done. Set info change ready to let
   * decoder continue decoding
   */
  ret = rk_api_->control(codec_ctx_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
  if (ret != MPP_OK) {
    auto msg = std::string("info change ready failed: ") + std::to_string(ret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  mpp_frame_deinit(&frame);
  frame = nullptr;
  return modelbox::STATUS_OK;
}

modelbox::Status RKNPUVideoDecoder::GetDecFrame(MppFrame &frame) {
  frame = nullptr;
  auto ret = rk_api_->decode_get_frame(codec_ctx_, &frame);
  if (MPP_ERR_TIMEOUT == ret) {
    MBLOG_DEBUG << "decode_get_frame failed too much time";
    return {modelbox::STATUS_FAULT, "decode_get_frame failed too much time"};
  }

  if (ret != MPP_OK) {
    MBLOG_ERROR << "decode_get_frame failed: " << ret;
    return {modelbox::STATUS_FAULT, "decode_get_frame failed"};
  }

  if (ret == MPP_OK && frame == nullptr) {
    // ok, no more, return STATUS_FAULT to exit while, not err here
    return modelbox::STATUS_FAULT;
  }

  if (mpp_frame_get_info_change(frame)) {
    return InfoChange(frame);
  }

  RK_U32 err_info = mpp_frame_get_errinfo(frame);
  if (err_info != MPP_OK) {
    err_number++;
    if (err_number % ERROR_LOG_TIMES == 0) {
      MBLOG_WARN << "frame error: " << err_info;
    }
    mpp_frame_deinit(&frame);
    frame = nullptr;
    // do not return STATUS_FAULT, just skip
  }

  return modelbox::STATUS_OK;
}

modelbox::Status RKNPUVideoDecoder::SendDecBuf(MppPacket &packet) {
  MPP_RET ret = MPP_OK;
  int times = DEC_RETRYS;

  while (running_ && times-- > 0) {
    ret = rk_api_->decode_put_packet(codec_ctx_, packet);
    if (MPP_OK == ret || MPP_ERR_BUFFER_FULL == ret) {
      break;
    }

    usleep(DEC_DELAY_TIMES);
  }

  if (MPP_OK == ret) {
    return modelbox::STATUS_OK;
  }

  if (MPP_ERR_BUFFER_FULL == ret) {
    usleep(RETRY_DELAY_TIMES);
    return modelbox::STATUS_AGAIN;
  }

  MBLOG_ERROR << "send decode frame fail: " << ret;
  return modelbox::STATUS_FAULT;
}

void RKNPUVideoDecoder::GetLimitDecFrame(std::vector<MppFrame> &out_frame,
                                         size_t max_frames) {
  MppFrame frame = nullptr;
  while (running_ && out_frame.size() < max_frames &&
         GetDecFrame(frame) == modelbox::STATUS_OK) {
    // inData=null 意味着是最后一帧， 全部获取完， 然后丢弃
    if (frame != nullptr) {
      out_frame.push_back(frame);
    }
  }
}

modelbox::Status RKNPUVideoDecoder::DecodeFrameBuf(
    const uint8_t *inData, size_t inSize, std::vector<MppFrame> &out_frame,
    size_t max_frames) {
  MppPacket packet;
  modelbox::Status ret = modelbox::STATUS_OK;
  auto mppret = mpp_packet_init(&packet, nullptr, 0);
  if (mppret != MPP_OK) {
    auto msg = std::string("mpp_packet_init failed: ") + std::to_string(mppret);
    MBLOG_ERROR << msg;
    return {modelbox::STATUS_FAULT, msg};
  }

  Defer { mpp_packet_deinit(&packet); };
  SetPacket(packet, (uint8_t *)inData, inSize);

  do {
    GetLimitDecFrame(out_frame, (inData == nullptr) ? INT_MAX : max_frames);
  } while ((ret = SendDecBuf(packet)) == modelbox::STATUS_AGAIN);

  // if last data, must get all frames, but send only max_frames to avoid block
  if (out_frame.size() > max_frames) {
    out_frame.resize(max_frames);
  }

  return ret;
}
