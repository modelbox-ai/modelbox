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

#include "nvcodec_video_decoder.h"

#include <string>

#include "modelbox/base/log.h"
#include "modelbox/device/cuda/device_cuda.h"

#define MIN_ALLOWABLE_DECODE_SURFACE_NUM 1

NvcodecConcurrencyLimiter *NvcodecConcurrencyLimiter::GetInstance() {
  static NvcodecConcurrencyLimiter limiter;
  return &limiter;
}

void NvcodecConcurrencyLimiter::Init(uint32_t limit) {
  if (limit == 0) {
    limited_ = false;
  }

  count_ = limit;
}

void NvcodecConcurrencyLimiter::Acquire() {
  if (!limited_) {
    return;
  }

  std::unique_lock<std::mutex> lock(count_lock_);
  count_cv_.wait(lock, [=] { return count_ > 0; });
  --count_;
}

void NvcodecConcurrencyLimiter::Release() {
  if (!limited_) {
    return;
  }

  std::unique_lock<std::mutex> lock(count_lock_);
  ++count_;
  count_cv_.notify_one();
}

#define NVDEC_THROW_ERROR(err_str, err_code)                                \
  throw NVDECException::MakeNVDECException(err_str, err_code, __FUNCTION__, \
                                           __FILE__, __LINE__);

NvcodecVideoDecoder::NvcodecVideoDecoder()
    : codec_id_map_{{AVCodecID::AV_CODEC_ID_MPEG1VIDEO,
                     cudaVideoCodec::cudaVideoCodec_MPEG1},
                    {AVCodecID::AV_CODEC_ID_MPEG2VIDEO,
                     cudaVideoCodec::cudaVideoCodec_MPEG2},
                    {AVCodecID::AV_CODEC_ID_MPEG4,
                     cudaVideoCodec::cudaVideoCodec_MPEG4},
                    {AVCodecID::AV_CODEC_ID_VC1,
                     cudaVideoCodec::cudaVideoCodec_VC1},
                    {AVCodecID::AV_CODEC_ID_H264,
                     cudaVideoCodec::cudaVideoCodec_H264},
                    {AVCodecID::AV_CODEC_ID_HEVC,
                     cudaVideoCodec::cudaVideoCodec_HEVC},
                    {AVCodecID::AV_CODEC_ID_VP8,
                     cudaVideoCodec::cudaVideoCodec_VP8},
                    {AVCodecID::AV_CODEC_ID_VP9,
                     cudaVideoCodec::cudaVideoCodec_VP9},
                    {AVCodecID::AV_CODEC_ID_MJPEG,
                     cudaVideoCodec::cudaVideoCodec_JPEG}},
      codec_id_name_map_{{cudaVideoCodec_MPEG1, "MPEG-1"},
                         {cudaVideoCodec_MPEG2, "MPEG-2"},
                         {cudaVideoCodec_MPEG4, "MPEG-4 (ASP)"},
                         {cudaVideoCodec_VP8, "VP8"},
                         {cudaVideoCodec_VP9, "VP9"},
                         {cudaVideoCodec_H264_SVC, "H.264/SVC"},
                         {cudaVideoCodec_H264_MVC, "H.264/MVC"},
                         {cudaVideoCodec_H264, "AVC/H.264"},
                         {cudaVideoCodec_VC1, "VC-1/WMV"},
                         {cudaVideoCodec_JPEG, "M-JPEG"},
                         {cudaVideoCodec_NV12, "NV12 4:2:0"},
                         {cudaVideoCodec_HEVC, "H.265/HEVC"},
                         {cudaVideoCodec_YUYV, "YUYV 4:2:2"},
                         {cudaVideoCodec_YV12, "YV12 4:2:0"},
                         {cudaVideoCodec_UYVY, "UYVY 4:2:2"},
                         {cudaVideoCodec_YUV420, "YUV  4:2:0"}} {}

NvcodecVideoDecoder::~NvcodecVideoDecoder() {
  auto ret = cudaSetDevice(gpu_id_);
  if (ret != cudaSuccess) {
    MBLOG_ERROR << "Set device to gpu " << gpu_id_ << " failed, err " << ret;
  }

  if (video_decoder_ != nullptr) {
    cuvidDestroyDecoder(video_decoder_);
    video_decoder_ = nullptr;
  }

  if (video_parser_ != nullptr) {
    cuvidDestroyVideoParser(video_parser_);
    video_parser_ = nullptr;
  }

  if (ctx_lock_ != nullptr) {
    cuvidCtxLockDestroy(ctx_lock_);
  }
}

modelbox::Status NvcodecVideoDecoder::Init(const std::string &device_id,
                                           AVCodecID codec_id,
                                           const std::string &file_url,
                                           bool skip_err_frame, bool no_delay) {
  gpu_id_ = std::stoi(device_id);
  MBLOG_INFO << "Init decode in gpu " << gpu_id_;
  // Use cuda runtime CUContext on same device in whole modelbox process to
  // ensure cuda work properly
  auto cuda_ret = cudaSetDevice(gpu_id_);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Set device to " << gpu_id_ << " failed, err " << cuda_ret;
    return modelbox::STATUS_FAULT;
  }

  CUcontext cu_ctx;
  auto cu_ret = cuCtxGetCurrent(&cu_ctx);
  if (cu_ret != CUDA_SUCCESS) {
    GET_CUDA_API_ERROR(cuCtxGetCurrent, cu_ret, err_str);
    MBLOG_ERROR << "Get Ctx in gpu " << gpu_id_ << " failed, err " << err_str;
    return modelbox::STATUS_FAULT;
  }

  cu_ret = cuvidCtxLockCreate(&ctx_lock_, cu_ctx);
  if (cu_ret != CUDA_SUCCESS) {
    GET_CUDA_API_ERROR(cuvidCtxLockCreate, cu_ret, err_str);
    MBLOG_ERROR << err_str << " : device " << device_id.c_str();
    return modelbox::STATUS_FAULT;
  }

  CUVIDPARSERPARAMS videoParserParams = {};
  auto ret = GetCudaVideoCodec(codec_id, codec_id_);
  if (ret != modelbox::STATUS_SUCCESS) {
    return ret;
  }

  videoParserParams.CodecType = codec_id_;
  videoParserParams.ulMaxNumDecodeSurfaces = 1;
  videoParserParams.ulMaxDisplayDelay =
      no_delay ? 0 : 2;  // setting ulMaxDisplayDelay to 2 achieves max decoding
                         // rate, based on several tests.
  videoParserParams.pUserData = (void *)this;
  videoParserParams.pfnSequenceCallback = HandleVideoSequenceProc;
  videoParserParams.pfnDecodePicture = HandlePictureDecodeProc;
  videoParserParams.pfnDisplayPicture = HandlePictureDisplayProc;

  cu_ret = cuvidCreateVideoParser(&video_parser_, &videoParserParams);
  if (cu_ret != CUDA_SUCCESS) {
    GET_CUDA_API_ERROR(cuvidCreateVideoParser, cu_ret, err_str);
    MBLOG_ERROR << err_str;
    return modelbox::STATUS_FAULT;
  }

  file_url_ = file_url;
  skip_err_frame_ = skip_err_frame;

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status NvcodecVideoDecoder::Decode(
    const std::shared_ptr<NvcodecPacket> &pkt,
    std::vector<std::shared_ptr<NvcodecFrame>> &frame_list, CUstream stream) {
  if (!video_parser_) {
    MBLOG_ERROR << "Nvcodec decode is not inited, parser is null";
    return modelbox::STATUS_FAULT;
  }

  CUVIDSOURCEDATAPACKET packet = {};
  packet.payload = pkt->GetDataRef();
  packet.payload_size = pkt->GetSize();
  packet.flags = CUVID_PKT_TIMESTAMP;
  packet.timestamp = pkt->GetPts();
  latest_pts_ = packet.timestamp;
  if (packet.payload == nullptr || packet.payload_size == 0) {
    packet.flags |= CUVID_PKT_ENDOFSTREAM;
  }

  video_stream_ = stream;
  frame_count_in_one_decode_ = 0;

  auto cuda_ret = cudaSetDevice(gpu_id_);
  if (cuda_ret != cudaSuccess) {
    MBLOG_ERROR << "Set device to gpu " << gpu_id_ << " failed, err "
                << cuda_ret;
    return modelbox::STATUS_FAULT;
  }

  NvcodecConcurrencyLimiter::GetInstance()->Acquire();
  is_limiter_released_ = false;
  CUDA_API_CALL(cuvidParseVideoData(video_parser_, &packet));
  if (!is_limiter_released_) {
    // might release when handle display
    NvcodecConcurrencyLimiter::GetInstance()->Release();
    is_limiter_released_ = true;
  }

  for (size_t i = 0; i < frame_count_in_one_decode_; ++i) {
    auto frame = std::make_shared<NvcodecFrame>();
    frame->data_ref = decoded_frame_buffer_list_[i].get();
    frame->width = GetWidth();
    frame->height = GetHeight();
    frame->timestamp = decoded_frame_timestamp_list_[i];
    frame_list.push_back(frame);
  }

  if (packet.flags & CUVID_PKT_ENDOFSTREAM) {
    return modelbox::STATUS_NODATA;
  }

  return modelbox::STATUS_SUCCESS;
}

modelbox::Status NvcodecVideoDecoder::GetCudaVideoCodec(
    AVCodecID codec_id, cudaVideoCodec &cuda_codec_id) {
  auto iter = codec_id_map_.find(codec_id);
  if (iter == codec_id_map_.end()) {
    MBLOG_ERROR << "ffmpeg code id[" << codec_id
                << "] for nvcodec is not supported";
    return modelbox::STATUS_NOTSUPPORT;
  }

  cuda_codec_id = iter->second;
  return modelbox::STATUS_SUCCESS;
}

std::string NvcodecVideoDecoder::GetVideoCodecString(
    cudaVideoCodec cuda_codec_id) {
  auto iter = codec_id_name_map_.find(cuda_codec_id);
  if (iter == codec_id_name_map_.end()) {
    return "Unknown";
  }

  return iter->second;
}

int32_t NvcodecVideoDecoder::HandleVideoSequence(CUVIDEOFORMAT *video_format) {
  uint32_t decode_surface = GetDecodeSurfaceNum(video_format);
  if (decode_surface <= 0) {
    decode_surface = MIN_ALLOWABLE_DECODE_SURFACE_NUM;
    MBLOG_WARN << "Invalid decode surface num ("
               << static_cast<int>(video_format->min_num_decode_surfaces)
               << "), change it to (" << MIN_ALLOWABLE_DECODE_SURFACE_NUM
               << ") for compatibility.";
  }

  CheckDeviceCaps(video_format);
  if (frame_width_ != 0 && luma_height_ != 0 && chroma_height_ != 0) {
    SequenceChanged(video_format);
    return decode_surface;
  }

  SaveSequenceParam(video_format);
  CreateDecoder(video_format, decode_surface);
  return decode_surface;
}

uint32_t NvcodecVideoDecoder::GetDecodeSurfaceNum(CUVIDEOFORMAT *video_format) {
  uint8_t num = 8;
  if (video_format->codec == cudaVideoCodec::cudaVideoCodec_VP9) {
    num = 12;
  } else if (video_format->codec == cudaVideoCodec::cudaVideoCodec_H264 ||
             video_format->codec == cudaVideoCodec::cudaVideoCodec_H264_SVC ||
             video_format->codec == cudaVideoCodec::cudaVideoCodec_H264_MVC ||
             video_format->codec == cudaVideoCodec::cudaVideoCodec_HEVC) {
    num = 20;
  }

  return std::max(video_format->min_num_decode_surfaces, num);
}

void NvcodecVideoDecoder::CheckDeviceCaps(CUVIDEOFORMAT *video_format) {
  CUVIDDECODECAPS decode_caps = {};
  decode_caps.eCodecType = video_format->codec;
  decode_caps.eChromaFormat = video_format->chroma_format;
  decode_caps.nBitDepthMinus8 = video_format->bit_depth_luma_minus8;

  CUDA_API_CALL(cuvidGetDecoderCaps(&decode_caps));
  if (!decode_caps.bIsSupported) {
    NVDEC_THROW_ERROR("Codec not supported on this GPU",
                      CUDA_ERROR_NOT_SUPPORTED);
  }

  if (video_format->coded_width > decode_caps.nMaxWidth ||
      video_format->coded_height > decode_caps.nMaxHeight) {
    std::ostringstream error_str;
    error_str << std::endl
              << "Resolution          : " << video_format->coded_width << " x "
              << video_format->coded_height << std::endl
              << "Max Supported (w x h) : " << decode_caps.nMaxWidth << " x "
              << decode_caps.nMaxHeight << std::endl
              << "Resolution not supported on this GPU";
    NVDEC_THROW_ERROR(error_str.str(), CUDA_ERROR_NOT_SUPPORTED);
  }

  if ((video_format->coded_width >> 4) * (video_format->coded_height >> 4) >
      decode_caps.nMaxMBCount) {
    std::ostringstream error_str;
    error_str << std::endl
              << "MBCount             : "
              << (video_format->coded_width >> 4) *
                     (video_format->coded_height >> 4)
              << std::endl
              << "Max Supported mbcnt : " << decode_caps.nMaxMBCount
              << std::endl
              << "MBCount not supported on this GPU";
    NVDEC_THROW_ERROR(error_str.str(), CUDA_ERROR_NOT_SUPPORTED);
  }
}

void NvcodecVideoDecoder::SequenceChanged(CUVIDEOFORMAT *video_format) {
  if (video_format_.coded_width == video_format->coded_width &&
      video_format_.coded_height == video_format->coded_height) {
    // No resolution change
    return;
  }

  NVDEC_THROW_ERROR("Resolution changed, decoded result may be incorrect",
                    CUDA_ERROR_ILLEGAL_STATE);
}

void NvcodecVideoDecoder::SaveSequenceParam(CUVIDEOFORMAT *video_format) {
  codec_id_ = video_format->codec;
  chroma_format_ = video_format->chroma_format;
  // Output format only supports YUV, so we only use nv12 here
  output_format_ = cudaVideoSurfaceFormat::cudaVideoSurfaceFormat_NV12;
  byte_depth_per_pixel_ = 1;
  video_format_ = *video_format;
}

void NvcodecVideoDecoder::CreateDecoder(CUVIDEOFORMAT *video_format,
                                        uint32_t decode_surface) {
  CUVIDDECODECREATEINFO decode_create_info = {};
  decode_create_info.CodecType = video_format->codec;
  decode_create_info.ChromaFormat = video_format->chroma_format;
  decode_create_info.OutputFormat = output_format_;
  decode_create_info.bitDepthMinus8 = video_format->bit_depth_luma_minus8;
  if (video_format->progressive_sequence) {
    decode_create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
  } else {
    decode_create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
  }
  decode_create_info.ulNumOutputSurfaces = 2;
  decode_create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  decode_create_info.ulNumDecodeSurfaces = decode_surface;
  decode_create_info.vidLock = ctx_lock_;
  decode_create_info.ulWidth = video_format->coded_width;
  decode_create_info.ulHeight = video_format->coded_height;
  decode_create_info.ulMaxWidth = video_format->coded_width;
  decode_create_info.ulMaxHeight = video_format->coded_height;

  frame_width_ =
      video_format->display_area.right - video_format->display_area.left;
  frame_height_ =
      video_format->display_area.bottom - video_format->display_area.top;
  decode_create_info.ulTargetWidth = video_format->coded_width;
  decode_create_info.ulTargetHeight = video_format->coded_height;

  luma_height_ = frame_height_;
  chroma_height_ = frame_height_ / 2;
  chroma_planes_number_ = 1;
  surface_height_ = decode_create_info.ulTargetHeight;
  surface_width_ = decode_create_info.ulTargetWidth;
  CUDA_API_CALL(cuvidCreateDecoder(&video_decoder_, &decode_create_info));
}

int32_t NvcodecVideoDecoder::HandlePictureDecode(CUVIDPICPARAMS *pic_params) {
  if (!video_decoder_) {
    NVDEC_THROW_ERROR("Decoder not init successed", CUDA_ERROR_NOT_INITIALIZED);
    return 0;
  }

  auto ret = cuvidDecodePicture(video_decoder_, pic_params);
  if (ret != CUDA_SUCCESS) {
    MBLOG_ERROR << "cuvidDecodePicture failed, ret: " << ret;
    return 0;
  }
  return 1;
}

int32_t NvcodecVideoDecoder::HandlePictureDisplay(
    CUVIDPARSERDISPINFO *display_info) {
  CUVIDPROCPARAMS proc_params = {};
  proc_params.progressive_frame = display_info->progressive_frame;
  proc_params.second_field = display_info->repeat_first_field + 1;
  proc_params.top_field_first = display_info->top_field_first;
  proc_params.unpaired_field = display_info->repeat_first_field < 0;
  proc_params.output_stream = video_stream_;

  CUdeviceptr src_frame_ptr = 0;
  uint32_t src_pitch = 0;
  CUDA_API_CALL(cuvidMapVideoFrame(video_decoder_, display_info->picture_index,
                                   &src_frame_ptr, &src_pitch, &proc_params));

  CUVIDGETDECODESTATUS decode_status = {};
  CUresult result = cuvidGetDecodeStatus(
      video_decoder_, display_info->picture_index, &decode_status);
  if (result == CUDA_SUCCESS &&
      (decode_status.decodeStatus == cuvidDecodeStatus_Error ||
       decode_status.decodeStatus == cuvidDecodeStatus_Error_Concealed)) {
    MBLOG_DEBUG << "Picture decode has error, image might be incorrect";
    if (skip_err_frame_) {
      CUDA_API_CALL(cuvidUnmapVideoFrame(video_decoder_, src_frame_ptr));
      return 1;
    }
  }

  if (latest_pts_ != 0 && display_info->timestamp > latest_pts_) {
    MBLOG_WARN << "Timestamp " << display_info->timestamp
               << " err, should not great than " << latest_pts_;
    CUDA_API_CALL(cuvidUnmapVideoFrame(video_decoder_, src_frame_ptr));
    return 1;
  }

  NvcodecConcurrencyLimiter::GetInstance()->Release();
  is_limiter_released_ = true;

  ++frame_count_in_one_decode_;
  SaveFrame(src_frame_ptr, src_pitch);
  SaveTimestamp(display_info->timestamp);
  CUDA_API_CALL(cuvidUnmapVideoFrame(video_decoder_, src_frame_ptr));
  return 1;
}

void NvcodecVideoDecoder::SaveFrame(CUdeviceptr src_frame_ptr,
                                    uint32_t src_pitch) {
  /* src frame is aligned, src pitch is great than frame width, so we need a mem
   * copy */
  uint8_t *decoded_frame_ptr =
      GetDecodeFramePtrFromCache(frame_count_in_one_decode_ - 1);
  CUDA_MEMCPY2D mem_cpy_2d = {};
  mem_cpy_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  mem_cpy_2d.srcDevice = src_frame_ptr;
  mem_cpy_2d.srcPitch = src_pitch;
  mem_cpy_2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  mem_cpy_2d.dstDevice = (CUdeviceptr)decoded_frame_ptr;
  mem_cpy_2d.dstPitch = frame_width_ * byte_depth_per_pixel_;
  mem_cpy_2d.WidthInBytes = frame_width_ * byte_depth_per_pixel_;
  mem_cpy_2d.Height = luma_height_;
  CUDA_API_CALL(cuMemcpy2D(&mem_cpy_2d));
  mem_cpy_2d.srcDevice = (CUdeviceptr)((uint8_t *)src_frame_ptr +
                                       mem_cpy_2d.srcPitch * surface_height_);
  mem_cpy_2d.dstDevice =
      (CUdeviceptr)(decoded_frame_ptr + mem_cpy_2d.dstPitch * luma_height_);
  mem_cpy_2d.Height = chroma_height_;
  CUDA_API_CALL(cuMemcpy2D(&mem_cpy_2d));
}

uint8_t *NvcodecVideoDecoder::GetDecodeFramePtrFromCache(size_t frame_index) {
  std::lock_guard<std::mutex> lock(decoded_frame_buffer_lock_);
  for (size_t i = decoded_frame_buffer_list_.size(); i <= frame_index; ++i) {
    // Not enough frames in buffer list, we need alloc one more
    uint8_t *frame_ptr = nullptr;
    CUDA_API_CALL(cuMemAlloc((CUdeviceptr *)&frame_ptr, GetFrameSize()));
    std::shared_ptr<uint8_t> frame_buffer(frame_ptr, [this](uint8_t *ptr) {
      if (this->ctx_mtx_ != nullptr) {
        this->ctx_mtx_->lock();
      }
      cuMemFree((CUdeviceptr)ptr);
      if (this->ctx_mtx_ != nullptr) {
        this->ctx_mtx_->unlock();
      }
    });
    decoded_frame_buffer_list_.push_back(frame_buffer);
  }

  return decoded_frame_buffer_list_[frame_index].get();
}

void NvcodecVideoDecoder::SaveTimestamp(int64_t timestamp) {
  if (decoded_frame_timestamp_list_.size() <
      decoded_frame_buffer_list_.size()) {
    decoded_frame_timestamp_list_.resize(decoded_frame_buffer_list_.size());
  }

  decoded_frame_timestamp_list_[frame_count_in_one_decode_ - 1] = timestamp;
}