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


#include "nppi_color_converter.h"
#include <modelbox/base/log.h>
#include <cuda.h>

NppiColorConverter::NppiColorConverter()
    : cvt_color_{{"rgb", nppiNV12ToRGB_8u_P2C3R},
                 {"bgr", nppiNV12ToBGR_8u_P2C3R}} {}

NppiColorConverter::~NppiColorConverter() = default;

modelbox::Status NppiColorConverter::CvtColor(const uint8_t *src, int32_t width,
                                            int32_t height, uint8_t *dest,
                                            const std::string &pix_fmt) {
  if (pix_fmt == "nv12") {
    auto cu_ret =
        cuMemcpy((CUdeviceptr)dest, (CUdeviceptr)src, width * height * 3 / 2);
    if (cu_ret != CUDA_SUCCESS) {
      MBLOG_ERROR << "cuMemcpy failed, ret " << cu_ret;
      return modelbox::STATUS_FAULT;
    }
  } else {
    auto iter = cvt_color_.find(pix_fmt);
    if (iter == cvt_color_.end()) {
      MBLOG_ERROR << "Not support pix_fmt " << pix_fmt;
      return modelbox::STATUS_NOTSUPPORT;
    }

    const uint8_t *src_arr[2];  // One for Y plane, one for UV plane
    src_arr[0] = src;
    src_arr[1] = src + width * height;
    auto ret = iter->second(src_arr, width, dest, width * 3, {width, height});
    if (ret != NPP_SUCCESS) {
      MBLOG_ERROR << "Cvt color from nv12 to " << pix_fmt << " failed, npp ret "
                  << ret;
      return modelbox::STATUS_FAULT;
    }
  }

  return modelbox::STATUS_SUCCESS;
}