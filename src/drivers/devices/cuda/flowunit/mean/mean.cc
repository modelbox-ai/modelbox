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


#include "mean.h"

#include "nppi_arithmetic_and_logical_operations.h"

bool CheckRoiValid(const ImageRect &roi) {
  if ((0 > roi.width) || (PIXEL_THRESHOLD < roi.width) || (0 > roi.height) ||
      (PIXEL_THRESHOLD < roi.height) || (0 > roi.x) ||
      (PIXEL_THRESHOLD < roi.x) || (0 > roi.y) || (PIXEL_THRESHOLD < roi.y)) {
    return false;
  }

  return true;
}

int32_t Mean_PLANAR_32f_P3R(const float *pSrcPlanarData, int width, int height,
                            const ImageRect &srcRoi, const ImageMean_32f &mean,
                            const cudaStream_t stream) {
  NppStatus status = NPP_ERROR;

  if (NULL == pSrcPlanarData) {
    MBLOG_ERROR << "Parma is Null.";
    return static_cast<int32_t>(status);
  }

  if ((0 > width) || (PIXEL_THRESHOLD < width)) {
    MBLOG_ERROR << "image width is invalid.";
    return static_cast<int32_t>(status);
  }

  if ((0 > height) || (PIXEL_THRESHOLD < height)) {
    MBLOG_ERROR << "image height is invalid.";
    return static_cast<int32_t>(status);
  }

  if ((0.0 > mean.channel_0) || (255.0 < mean.channel_1) ||
      (0.0 > mean.channel_2) || (255.0 < mean.channel_0) ||
      (0.0 > mean.channel_1) || (255.0 < mean.channel_2)) {
    MBLOG_ERROR << "mean value is invalid.";
    return static_cast<int32_t>(status);
  }

  if (!CheckRoiValid(srcRoi)) {
    return static_cast<int32_t>(status);
  }

  NppiSize oSizeROI;
  oSizeROI.width = srcRoi.width;
  oSizeROI.height = srcRoi.height;

  status = nppiSubC_32f_C1R((Npp32f *)pSrcPlanarData, width * sizeof(float),
                            mean.channel_0, (Npp32f *)pSrcPlanarData,
                            width * sizeof(float), oSizeROI);
  if (NPP_SUCCESS != status) {
    return static_cast<int32_t>(status);
  }

  status = nppiSubC_32f_C1R((Npp32f *)pSrcPlanarData + height * width,
                            width * sizeof(float), mean.channel_1,
                            (Npp32f *)pSrcPlanarData + height * width,
                            width * sizeof(float), oSizeROI);
  if (NPP_SUCCESS != status) {
    return static_cast<int32_t>(status);
  }

  status = nppiSubC_32f_C1R((Npp32f *)pSrcPlanarData + height * width * 2,
                            width * sizeof(float), mean.channel_2,
                            (Npp32f *)pSrcPlanarData + height * width * 2,
                            width * sizeof(float), oSizeROI);

  return static_cast<int32_t>(status);
}