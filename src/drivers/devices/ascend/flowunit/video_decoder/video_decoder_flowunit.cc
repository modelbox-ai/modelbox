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

#include "video_decoder_flowunit.h"

#include <securec.h>

#include <fstream>
#include <string>

#include "ascend_video_decode.h"
#include "modelbox/base/timer.h"
#include "modelbox/base/utils.h"
#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

VideoDecodeFlowUnit::VideoDecodeFlowUnit() = default;
VideoDecodeFlowUnit::~VideoDecodeFlowUnit() = default;

static std::map<std::string, int32_t> fmt_trans_map = {
    {"nv12", PIXEL_FORMAT_YUV_SEMIPLANAR_420}};

constexpr int32_t PROFILE_BASELINE = 66;
constexpr int32_t PROFILE_MAIN = 77;
constexpr int32_t PROFILE_HIGH = 100;
constexpr int32_t PROFILE_DEFAULT = -1;

static std::unordered_map<AVCodecID, std::unordered_map<int32_t, int32_t>>
    encode_type_map = {{AV_CODEC_ID_HEVC, {{PROFILE_DEFAULT, H265_MAIN_LEVEL}}},
                       {AV_CODEC_ID_H264,
                        {{PROFILE_BASELINE, H264_BASELINE_LEVEL},
                         {PROFILE_MAIN, H264_MAIN_LEVEL},
                         {PROFILE_HIGH, H264_HIGH_LEVEL},
                         {PROFILE_DEFAULT, H264_MAIN_LEVEL}}}};

int32_t VideoDecodeFlowUnit::GetDvppEncodeType(AVCodecID codec_id,
                                               int32_t profile_id) {
  auto codec_item = encode_type_map.find(codec_id);
  if (codec_item == encode_type_map.end()) {
    MBLOG_ERROR << "Not support codec id " << codec_id;
    return -1;
  }

  auto &codec_profile_map = codec_item->second;
  auto profile_item = codec_profile_map.find(profile_id);
  if (profile_item == codec_profile_map.end()) {
    return codec_profile_map[PROFILE_DEFAULT];
  }

  return profile_item->second;
}

modelbox::Status VideoDecodeFlowUnit::GetDecoderParam(
    const std::shared_ptr<modelbox::DataContext> &data_ctx, int32_t &rate_num,
    int32_t &rate_den, int32_t &encode_type) {
  auto input_packet = data_ctx->Input(VIDEO_PACKET_INPUT);
  if (input_packet == nullptr) {
    return {modelbox::STATUS_FAULT, "get input failed."};
  }

  auto buffer = input_packet->At(0);
  auto res = buffer->Get("rate_num", rate_num);
  if (!res) {
    return {modelbox::STATUS_FAULT, "get rate_num failed."};
  }

  res = buffer->Get("rate_den", rate_den);
  if (!res) {
    return {modelbox::STATUS_FAULT, "get rate_den failed."};
  }

  auto in_meta = data_ctx->GetInputMeta(VIDEO_PACKET_INPUT);
  auto codec_id =
      std::static_pointer_cast<AVCodecID>(in_meta->GetMeta(CODEC_META));
  if (codec_id == nullptr) {
    return {modelbox::STATUS_FAULT, "get codec id failed."};
  }

  auto profile_id =
      std::static_pointer_cast<int32_t>(in_meta->GetMeta(PROFILE_META));
  if (profile_id == nullptr) {
    return {modelbox::STATUS_FAULT, "get profile id failed."};
  }

  encode_type = GetDvppEncodeType(*codec_id, *profile_id);
  if (encode_type == -1) {
    return {modelbox::STATUS_FAULT, "get dvpp encode type failed."};
  }

  return modelbox::STATUS_OK;
}

constexpr int32_t MAX_VDEC_CHAN = 128;

void VideoDecodeFlowUnit::InitInstanceId() {
  for (int i = 0; i < MAX_VDEC_CHAN; i++) {
    instance_available_map_[i] = true;
  }
}

int32_t VideoDecodeFlowUnit::FindTheMinimumAvailableId() {
  std::lock_guard<std::mutex> lk(mutex);
  for (auto &instance_item : instance_available_map_) {
    {
      if (instance_item.second) {
        instance_item.second = false;
        return instance_item.first;
      }
    }
  }

  return -1;
}

void VideoDecodeFlowUnit::RestoreInstanceId(int32_t instance_id) {
  std::lock_guard<std::mutex> lk(mutex);
  instance_available_map_[instance_id] = true;
}

modelbox::Status VideoDecodeFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  std::string fmt = opts->GetString("pix_fmt", "nv12");

  auto iter = fmt_trans_map.find(fmt);
  if (iter == fmt_trans_map.end()) {
    MBLOG_ERROR << "Not support pix fmt " << fmt;
    return modelbox::STATUS_BADCONF;
  }

  format_ = fmt_trans_map[fmt];

  InitInstanceId();

  return modelbox::STATUS_OK;
}

modelbox::Status VideoDecodeFlowUnit::ReopenDecoder(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::shared_ptr<modelbox::Buffer> &flag_buffer) {
  auto old_source_url =
      std::static_pointer_cast<std::string>(data_ctx->GetPrivate(SOURCE_URL_META));
  auto old_codec_id =
      std::static_pointer_cast<AVCodecID>(data_ctx->GetPrivate(CODEC_ID_META));

  if (old_source_url == nullptr || old_codec_id == nullptr) {
    MBLOG_ERROR << "Reopen decoder failed, source url or codec id is null";
    return modelbox::STATUS_FAULT;
  }

  std::string source_url;
  AVCodecID codec_id;
  if (flag_buffer->Get(SOURCE_URL_META, source_url) == false) {
    return modelbox::STATUS_SUCCESS;
  }

  if (flag_buffer->Get(CODEC_ID_META, codec_id) == false) {
    return modelbox::STATUS_SUCCESS;
  }

  if (source_url == *old_source_url && codec_id == *old_codec_id) {
    return modelbox::STATUS_SUCCESS;
  }

  MBLOG_WARN << "Reopen decoder, source url or codec id changed";
  auto ret = CloseDecoder(data_ctx);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Close decoder failed";
    return modelbox::STATUS_FAULT;
  }

  int32_t rate_num;
  int32_t rate_den;
  int32_t encode_type;

  auto res = flag_buffer->Get("rate_num", rate_num);
  if (!res) {
    return {modelbox::STATUS_FAULT, "get rate_num failed."};
  }

  res = flag_buffer->Get("rate_den", rate_den);
  if (!res) {
    return {modelbox::STATUS_FAULT, "get rate_den failed."};
  }

  auto in_meta = data_ctx->GetInputMeta(VIDEO_PACKET_INPUT);
  auto profile_id =
      std::static_pointer_cast<int32_t>(in_meta->GetMeta(PROFILE_META));
  if (profile_id == nullptr) {
    return {modelbox::STATUS_FAULT, "get profile id failed."};
  }

  encode_type = GetDvppEncodeType(codec_id, *profile_id);
  if (encode_type == -1) {
    return {modelbox::STATUS_FAULT, "get dvpp encode type failed."};
  }

  return NewDecoder(data_ctx, source_url, codec_id, rate_num, rate_den,
                    encode_type);
}

modelbox::Status VideoDecodeFlowUnit::CloseDecoder(
    std::shared_ptr<modelbox::DataContext> &data_ctx) {
  auto instance_id =
      std::static_pointer_cast<int32_t>(data_ctx->GetPrivate(INSTANCE_ID));
  if (instance_id != nullptr) {
    RestoreInstanceId(*instance_id);
  }
  data_ctx->SetPrivate(DVPP_DECODER, nullptr);
  data_ctx->SetPrivate(DVPP_DECODER_CTX, nullptr);
  data_ctx->SetPrivate(FRAME_INDEX_CTX, nullptr);
  data_ctx->SetPrivate(INSTANCE_ID, nullptr);
  data_ctx->SetPrivate(SOURCE_URL_META, nullptr);
  data_ctx->SetPrivate(CODEC_ID_META, nullptr);
  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecodeFlowUnit::NewDecoder(
    std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::string &source_url, AVCodecID codec_id, int32_t rate_num,
    int32_t rate_den, int32_t encode_type) {
  int32_t instance_id = 0;
  instance_id = FindTheMinimumAvailableId();
  modelbox::Status ret = modelbox::STATUS_SUCCESS;
  DeferCond { return !ret; };

  if (instance_id == -1) {
    const auto *errMsg = "do not have available channelId to decode.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  DeferCondAdd { RestoreInstanceId(instance_id); };

  auto video_decoder = std::make_shared<AscendVideoDecoder>(
      instance_id, dev_id_, rate_num, rate_den, format_, encode_type);
  ret = video_decoder->Init(data_ctx);
  if (ret != modelbox::STATUS_SUCCESS) {
    auto errMsg = "video decoder init failed, " + ret.WrapErrormsgs();
    MBLOG_ERROR << errMsg;
    ret = {modelbox::STATUS_FAULT, errMsg};
    return ret;
  }

  auto dvpp_decode_ctx = std::make_shared<DvppVideoDecodeContext>();

  auto frame_index = std::make_shared<int64_t>();
  *frame_index = 0;
  auto instance_id_ptr = std::make_shared<int32_t>(instance_id);
  data_ctx->SetPrivate(DVPP_DECODER_CTX, dvpp_decode_ctx);
  data_ctx->SetPrivate(DVPP_DECODER, video_decoder);
  data_ctx->SetPrivate(FRAME_INDEX_CTX, frame_index);
  data_ctx->SetPrivate(INSTANCE_ID, instance_id_ptr);
  data_ctx->SetPrivate(SOURCE_URL_META, std::make_shared<std::string>(source_url));
  data_ctx->SetPrivate(CODEC_ID_META, std::make_shared<AVCodecID>(codec_id));
  MBLOG_INFO << "open video decode data success.";

  return ret;
}

modelbox::Status VideoDecodeFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto input_packet = data_ctx->Input(VIDEO_PACKET_INPUT);
  if (input_packet == nullptr) {
    return {modelbox::STATUS_FAULT, "get input failed."};
  }

  int32_t rate_num;
  int32_t rate_den;
  int32_t encode_type;

  auto buffer = input_packet->At(0);
  auto res = buffer->Get("rate_num", rate_num);
  if (!res) {
    return {modelbox::STATUS_FAULT, "get rate_num failed."};
  }

  res = buffer->Get("rate_den", rate_den);
  if (!res) {
    return {modelbox::STATUS_FAULT, "get rate_den failed."};
  }

  auto in_meta = data_ctx->GetInputMeta(VIDEO_PACKET_INPUT);
  auto codec_id =
      std::static_pointer_cast<AVCodecID>(in_meta->GetMeta(CODEC_META));
  if (codec_id == nullptr) {
    return {modelbox::STATUS_FAULT, "get codec id failed."};
  }

  auto source_url =
      std::static_pointer_cast<std::string>(in_meta->GetMeta(SOURCE_URL_META));
  if (source_url == nullptr) {
    MBLOG_ERROR << "Stream source url is null, init decoder failed";
    return modelbox::STATUS_FAULT;
  }

  auto profile_id =
      std::static_pointer_cast<int32_t>(in_meta->GetMeta(PROFILE_META));
  if (profile_id == nullptr) {
    return {modelbox::STATUS_FAULT, "get profile id failed."};
  }

  encode_type = GetDvppEncodeType(*codec_id, *profile_id);
  if (encode_type == -1) {
    return {modelbox::STATUS_FAULT, "get dvpp encode type failed."};
  }

  return NewDecoder(data_ctx, *source_url, *codec_id, rate_num, rate_den,
                    encode_type);
};

modelbox::Status VideoDecodeFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  return CloseDecoder(data_ctx);
}

modelbox::Status VideoDecodeFlowUnit::Close() {
  instance_available_map_.clear();
  return modelbox::STATUS_OK;
}

modelbox::Status VideoDecodeFlowUnit::ReadData(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    std::vector<std::shared_ptr<DvppPacket>> &dvpp_packet_list,
    std::shared_ptr<modelbox::Buffer> &flag_buffer) {
  auto reset_flag = false;
  auto video_packet_input = data_ctx->Input(VIDEO_PACKET_INPUT);
  if (video_packet_input == nullptr) {
    MBLOG_ERROR << "video packet input is null";
    return modelbox::STATUS_FAULT;
  }

  if (video_packet_input->Size() == 0) {
    MBLOG_ERROR << "video packet input size is 0";
    return modelbox::STATUS_FAULT;
  }

  for (size_t i = 0; i < video_packet_input->Size(); ++i) {
    auto packet_buffer = video_packet_input->At(i);

    if (reset_flag == false) {
      packet_buffer->Get("reset_flag", reset_flag);
      if (reset_flag == true) {
        flag_buffer = packet_buffer;
      }
    }

    std::shared_ptr<DvppPacket> dvpp_packet;
    auto ret = ReadDvppStreamDesc(packet_buffer, dvpp_packet);
    if (ret != modelbox::STATUS_SUCCESS) {
      auto errMsg = "read dvpp stream desc " + ret.WrapErrormsgs();
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }

    dvpp_packet_list.push_back(dvpp_packet);
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecodeFlowUnit::SetUpTheLastPacket(
    std::shared_ptr<DvppPacket> &dvpp_packet) {
  dvpp_packet = std::make_shared<DvppPacket>();
  dvpp_packet->SetEnd(true);

  auto *dvpp_stream_desc_ptr = acldvppCreateStreamDesc();
  if (dvpp_stream_desc_ptr == nullptr) {
    const auto *errMsg = "fail to create input stream desc";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto ret = acldvppSetStreamDescEos(dvpp_stream_desc_ptr, 1);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg =
        "fail to set data for stream desc, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    auto des_ret = acldvppDestroyStreamDesc(dvpp_stream_desc_ptr);
    if (des_ret != ACL_ERROR_NONE) {
      MBLOG_ERROR << "fail to destroy input stream desc";
    }

    return {modelbox::STATUS_FAULT, errMsg};
  }

  dvpp_packet->SetStreamDesc(dvpp_stream_desc_ptr);

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecodeFlowUnit::ReadDvppStreamDesc(
    const std::shared_ptr<modelbox::Buffer> &packet_buffer,
    std::shared_ptr<DvppPacket> &dvpp_packet) {
  auto size = packet_buffer->GetBytes();
  if (size == 1) {
    auto status = SetUpTheLastPacket(dvpp_packet);
    if (status != modelbox::STATUS_SUCCESS) {
      auto errMsg = "setup the last packet failed, " + status.WrapErrormsgs();
      MBLOG_ERROR << errMsg;
      return {modelbox::STATUS_FAULT, errMsg};
    }

    return status;
  }

  const auto *buffer = packet_buffer->ConstData();
  int32_t width = 0;
  int32_t height = 0;
  int64_t pts = 0;
  auto exists = packet_buffer->Get("width", width);
  if (!exists) {
    const auto *errMsg = "get width in input buffer meta failed.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  exists = packet_buffer->Get("height", height);
  if (!exists) {
    const auto *errMsg = "get width in input buffer meta failed.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  exists = packet_buffer->Get("pts", pts);
  if (!exists) {
    const auto *errMsg = "get pts in input buffer meta failed.";
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  dvpp_packet = std::make_shared<DvppPacket>(size, width, height, pts);

  void *temp_ptr = nullptr;
  bool dvpp_alloc_result = false;
  DeferCond { return dvpp_alloc_result; };

  auto ret = acldvppMalloc(&temp_ptr, size);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg = "acldvppMalloc failed, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  DeferCondAdd {
    if (temp_ptr != nullptr) {
      acldvppFree(temp_ptr);
    }
    temp_ptr = nullptr;
  };

  ret = aclrtMemcpy(temp_ptr, size, buffer, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg = "fail to memory copy, err code" + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    dvpp_alloc_result = true;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  auto *dvpp_stream_desc_ptr = acldvppCreateStreamDesc();
  if (dvpp_stream_desc_ptr == nullptr) {
    const auto *errMsg = "fail to create input stream desc";
    MBLOG_ERROR << errMsg;
    dvpp_alloc_result = true;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  DeferCondAdd {
    ret = acldvppDestroyStreamDesc(dvpp_stream_desc_ptr);
    if (ret != ACL_ERROR_NONE) {
      MBLOG_ERROR << "destroy stream desc failed, err code " << ret;
    }
  };

  ret = acldvppSetStreamDescData(dvpp_stream_desc_ptr, temp_ptr);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg =
        "fail to set data for stream desc, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    dvpp_alloc_result = true;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  // set size for dvpp stream desc
  ret = acldvppSetStreamDescSize(dvpp_stream_desc_ptr, size);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg =
        "fail to set size for stream desc, err code " + std::to_string(ret);
    MBLOG_ERROR << errMsg;
    dvpp_alloc_result = true;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  ret = acldvppSetStreamDescTimestamp(dvpp_stream_desc_ptr, (uint64_t)pts);
  if (ret != ACL_ERROR_NONE) {
    auto errMsg = "fail to set size for stream time stamp, err code " +
                  std::to_string(ret);
    MBLOG_ERROR << errMsg;
    dvpp_alloc_result = true;
    return {modelbox::STATUS_FAULT, errMsg};
  }

  dvpp_packet->SetStreamDesc(dvpp_stream_desc_ptr);

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecodeFlowUnit::WriteData(
    const std::shared_ptr<modelbox::DataContext> &data_ctx,
    const std::shared_ptr<AscendVideoDecoder> &video_decoder,
    const std::shared_ptr<DvppVideoDecodeContext> &dvpp_ctx) {
  auto queue = dvpp_ctx->GetCacheQueue();
  size_t size;

  auto output_bufs = data_ctx->Output(FRAME_INFO_OUTPUT);
  std::vector<std::shared_ptr<DvppFrame>> dvpp_frame;
  size = queue->PopBatch(&dvpp_frame, -1);

  if (size == 0) {
    return modelbox::STATUS_SUCCESS;
  }

  auto frame_index =
      std::static_pointer_cast<int64_t>(data_ctx->GetPrivate(FRAME_INDEX_CTX));
  auto rate_num = video_decoder->GetRateNum();
  auto rate_den = video_decoder->GetRateDen();

  auto device = this->GetBindDevice();
  for (size_t i = 0; i < size; ++i) {
    auto *pic_desc = dvpp_frame[i]->GetPicDesc().get();
    void *data = acldvppGetPicDescData(pic_desc);
    if (data == nullptr) {
      MBLOG_ERROR << "output pic data is nullptr.";
      continue;
    }

    uint32_t data_size = acldvppGetPicDescSize(pic_desc);
    if (data_size == 0) {
      acldvppFree(data);
      MBLOG_ERROR << "output pic data size is 0.";
      continue;
    }

    std::shared_ptr<modelbox::Buffer> buffer =
        std::make_shared<modelbox::Buffer>(device, modelbox::ASCEND_MEM_DVPP);
    buffer->Build(data, data_size, acldvppFree);

    auto width = acldvppGetPicDescWidth(pic_desc);
    auto height = acldvppGetPicDescHeight(pic_desc);
    auto width_stride = acldvppGetPicDescWidthStride(pic_desc);
    auto height_stride = acldvppGetPicDescHeightStride(pic_desc);

    buffer->Set("width", (int)width);
    buffer->Set("height", (int)height);
    buffer->Set("width_stride", (int)width_stride);
    buffer->Set("height_stride", (int)height_stride);
    buffer->Set("pix_fmt", std::string(OUTPUT_PIX_FMT));
    buffer->Set("channel", (int32_t)1);
    buffer->Set("shape", std::vector<size_t>{(size_t)height_stride * 3 / 2,
                                             (size_t)width_stride, 1});
    buffer->Set("layout", std::string("hwc"));
    buffer->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
    buffer->Set("index", *frame_index);
    *frame_index = *frame_index + 1;
    buffer->Set("rate_num", rate_num);
    buffer->Set("rate_den", rate_den);
    buffer->Set("eos", false);

    output_bufs->PushBack(buffer);
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status VideoDecodeFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  std::shared_ptr<modelbox::Buffer> flag_buffer = nullptr;

  auto acl_ret = aclrtSetDevice(dev_id_);
  if (acl_ret != ACL_SUCCESS) {
    MBLOG_ERROR << "set acl device to " << dev_id_ << " failed, err "
                << acl_ret;
    return modelbox::STATUS_FAULT;
  }

  auto video_decoder_ctx = std::static_pointer_cast<DvppVideoDecodeContext>(
      data_ctx->GetPrivate(DVPP_DECODER_CTX));
  auto video_decoder = std::static_pointer_cast<AscendVideoDecoder>(
      data_ctx->GetPrivate(DVPP_DECODER));
  if (video_decoder == nullptr || video_decoder_ctx == nullptr) {
    MBLOG_ERROR << "Video decoder is not init";
    return modelbox::STATUS_FAULT;
  }

  auto ret = WriteData(data_ctx, video_decoder, video_decoder_ctx);
  if (ret != modelbox::STATUS_SUCCESS) {
    return modelbox::STATUS_FAULT;
  }

  auto event = data_ctx->Event();
  if (event != nullptr) {
    return modelbox::STATUS_CONTINUE;
  }

  std::vector<std::shared_ptr<DvppPacket>> dvpp_packet_list;
  ret = ReadData(data_ctx, dvpp_packet_list, flag_buffer);
  if (ret != modelbox::STATUS_SUCCESS) {
    MBLOG_ERROR << "Read av_packet input failed, err code " + ret.ToString();
    return modelbox::STATUS_FAULT;
  }

  if (flag_buffer) {
    video_decoder_ctx = nullptr;
    video_decoder = nullptr;
    if (ReopenDecoder(data_ctx, flag_buffer) != modelbox::STATUS_SUCCESS) {
      MBLOG_ERROR << "Reopen decoder failed";
      return modelbox::STATUS_FAULT;
    }

    video_decoder_ctx = std::static_pointer_cast<DvppVideoDecodeContext>(
        data_ctx->GetPrivate(DVPP_DECODER_CTX));
    video_decoder = std::static_pointer_cast<AscendVideoDecoder>(
        data_ctx->GetPrivate(DVPP_DECODER));
    if (video_decoder == nullptr || video_decoder_ctx == nullptr) {
      MBLOG_ERROR << "Video decoder is not init";
      return modelbox::STATUS_FAULT;
    }
  }

  size_t err_num = 0;
  modelbox::Status decode_ret = modelbox::STATUS_SUCCESS;
  for (auto &dvpp_pkt : dvpp_packet_list) {
    int retry_num = 0;
    do {
      decode_ret = video_decoder->Decode(dvpp_pkt, video_decoder_ctx);
      if (decode_ret == modelbox::STATUS_FAULT) {
        MBLOG_ERROR << "video decoder a packet failed, "
                    << decode_ret.WrapErrormsgs();
        retry_num++;
      }
    } while (retry_num <= DECODER_RETRY_NUM &&
             decode_ret == modelbox::STATUS_FAULT);

    if (decode_ret == modelbox::STATUS_FAULT) {
      err_num++;
    }
  }

  if (err_num == dvpp_packet_list.size()) {
    return {modelbox::STATUS_FAULT, "video decoder failed."};
  }

  if (decode_ret == modelbox::STATUS_NODATA) {
    MBLOG_INFO << "write the last frame. ";
    ret = WriteData(data_ctx, video_decoder, video_decoder_ctx);
    if (ret != modelbox::STATUS_SUCCESS) {
      MBLOG_ERROR << "Write the last frame failed";
      return modelbox::STATUS_FAULT;
    }
  }

  return modelbox::STATUS_SUCCESS;
}

MODELBOX_FLOWUNIT(VideoDecodeFlowUnit, desc) {
  desc.SetFlowUnitName(DVPP_DECODE_FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Video");
  desc.AddFlowUnitInput({VIDEO_PACKET_INPUT, "cpu"});
  desc.AddFlowUnitOutput({FRAME_INFO_OUTPUT, modelbox::ASCEND_MEM_DVPP});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetInputContiguous(false);
  desc.SetResourceNice(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption("pix_fmt", "string", true,
                                                  "nv12", "the pix format"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
