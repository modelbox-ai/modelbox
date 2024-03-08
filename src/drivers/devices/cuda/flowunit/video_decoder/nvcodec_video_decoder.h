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

#ifndef MODELBOX_FLOWUNIT_NVCODEC_VIDEO_DECODER_H_
#define MODELBOX_FLOWUNIT_NVCODEC_VIDEO_DECODER_H_

#include <cuda.h>
#include <libavformat/avformat.h>
#include <modelbox/base/status.h>
#include <nvcuvid.h>

#include <condition_variable>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

class NvcodecConcurrencyLimiter {
 public:
  static NvcodecConcurrencyLimiter *GetInstance();

  void Init(uint32_t limit);

  void Acquire();

  void Release();

 private:
  NvcodecConcurrencyLimiter() = default;

  std::mutex count_lock_;
  std::condition_variable count_cv_;
  uint32_t count_{0};
  bool limited_{false};
};

class NVDECException : public std::exception {
 public:
  NVDECException(std::string err_str, const CUresult err_code)
      : err_str_(std::move(err_str)), err_code_(err_code) {}

  ~NVDECException() noexcept override = default;
  const char *what() const noexcept override { return err_str_.c_str(); }
  CUresult GetErrorCode() const { return err_code_; }
  const std::string &GetErrorString() const { return err_str_; }
  static NVDECException MakeNVDECException(const std::string &err_str,
                                           CUresult err_code,
                                           const std::string &function_name,
                                           const std::string &file_name,
                                           int line_number);

 private:
  std::string err_str_;
  CUresult err_code_;
};

inline NVDECException NVDECException::MakeNVDECException(
    const std::string &err_str, const CUresult err_code,
    const std::string &function_name, const std::string &file_name,
    int line_number) {
  std::ostringstream error_log;
  error_log << function_name << " : " << err_str << " at " << file_name << ":"
            << line_number << std::endl;
  NVDECException exception(error_log.str(), err_code);
  return exception;
}

class NvcodecFrame {
 public:
  int32_t width{0};
  int32_t height{0};
  int64_t timestamp{0};
  uint8_t *data_ref{nullptr};
};

class NvcodecPacket {
 public:
  NvcodecPacket(size_t size, const uint8_t *data_ref, int64_t pts)
      : size_(size), data_ref_(data_ref), pts_(pts) {}

  NvcodecPacket() = default;

  virtual ~NvcodecPacket() = default;

  size_t GetSize() { return size_; };

  const uint8_t *GetDataRef() { return data_ref_; };

  int64_t GetPts() { return pts_; };

 private:
  size_t size_{0};
  const uint8_t *data_ref_{nullptr};
  int64_t pts_{0};
};

class NvcodecVideoDecoder {
 public:
  NvcodecVideoDecoder();

  virtual ~NvcodecVideoDecoder();

  modelbox::Status Init(const std::string &device_id, AVCodecID codec_id,
                        const std::string &file_url, bool skip_err_frame,
                        bool no_delay);

  modelbox::Status Decode(
      const std::shared_ptr<NvcodecPacket> &pkt,
      std::vector<std::shared_ptr<NvcodecFrame>> &frame_list,
      CUstream stream = nullptr);

  int32_t GetWidth() { return frame_width_; }

  int32_t GetHeight() { return frame_height_; }

  const std::string &GetFileUrl() { return file_url_; }

 private:
  modelbox::Status InitCuCtx(const std::string &device_id);

  modelbox::Status GetCudaVideoCodec(AVCodecID codec_id,
                                     cudaVideoCodec &cuda_codec_id);

  std::string GetVideoCodecString(cudaVideoCodec cuda_codec_id);

  static int32_t CUDAAPI HandleVideoSequenceProc(void *user_data,
                                                 CUVIDEOFORMAT *video_format) {
    return ((NvcodecVideoDecoder *)user_data)
        ->HandleVideoSequence(video_format);
  }

  static int32_t CUDAAPI HandlePictureDecodeProc(void *user_data,
                                                 CUVIDPICPARAMS *pic_params) {
    return ((NvcodecVideoDecoder *)user_data)->HandlePictureDecode(pic_params);
  }

  static int32_t CUDAAPI
  HandlePictureDisplayProc(void *user_data, CUVIDPARSERDISPINFO *display_info) {
    return ((NvcodecVideoDecoder *)user_data)
        ->HandlePictureDisplay(display_info);
  }

  int32_t HandleVideoSequence(CUVIDEOFORMAT *video_format);

  void CheckDeviceCaps(CUVIDEOFORMAT *video_format);

  void SequenceChanged(CUVIDEOFORMAT *video_format);

  void SaveSequenceParam(CUVIDEOFORMAT *video_format);

  void CreateDecoder(CUVIDEOFORMAT *video_format, uint32_t decode_surface);

  int32_t HandlePictureDecode(CUVIDPICPARAMS *pic_params);

  int32_t HandlePictureDisplay(CUVIDPARSERDISPINFO *display_info);

  void SaveFrame(CUdeviceptr src_frame_ptr, uint32_t src_pitch);

  uint8_t *GetDecodeFramePtrFromCache(size_t frame_index);

  void SaveTimestamp(int64_t timestamp);

  inline int32_t GetFrameSize() {
    return frame_width_ * (luma_height_ + chroma_height_);
  }

  uint32_t GetDecodeSurfaceNum(CUVIDEOFORMAT *video_format);

  CUvideoparser video_parser_{nullptr};
  CUstream video_stream_{nullptr};
  CUcontext ctx_{nullptr};
  std::mutex *ctx_mtx_{nullptr};
  CUvideoctxlock ctx_lock_{nullptr};
  CUvideodecoder video_decoder_{nullptr};
  std::map<AVCodecID, cudaVideoCodec> codec_id_map_;
  std::map<cudaVideoCodec, std::string> codec_id_name_map_;

  int32_t frame_width_{0};
  int32_t frame_height_{0};
  int32_t luma_height_{0};
  int32_t chroma_height_{0};
  int32_t chroma_planes_number_{0};
  int32_t surface_height_{0};
  int32_t surface_width_{0};
  cudaVideoChromaFormat chroma_format_{};
  cudaVideoSurfaceFormat output_format_{};
  uint8_t byte_depth_per_pixel_{1};
  cudaVideoCodec codec_id_{};
  CUVIDEOFORMAT video_format_{};

  std::mutex decoded_frame_buffer_lock_;
  size_t frame_count_in_one_decode_{0};
  std::vector<std::shared_ptr<uint8_t>> decoded_frame_buffer_list_;
  std::vector<int64_t> decoded_frame_timestamp_list_;

  std::string file_url_;
  bool skip_err_frame_{false};
  int64_t latest_pts_{0};
  int32_t gpu_id_{0};

  bool is_limiter_released_{false};
};

#endif  // MODELBOX_FLOWUNIT_NVCODEC_VIDEO_DECODER_H_
