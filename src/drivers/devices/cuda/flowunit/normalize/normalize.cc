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


#include "normalize.h"

#include "nppi_arithmetic_and_logical_operations.h"

bool CheckRoiValid(const ImageRect &roi) {
  if ((0 > roi.width) || (PIXEL_THRESHOLD < roi.width) || (0 > roi.height) ||
      (PIXEL_THRESHOLD < roi.height) || (0 > roi.x) ||
      (PIXEL_THRESHOLD < roi.x) || (0 > roi.y) || (PIXEL_THRESHOLD < roi.y)) {
    return false;
  }

  return true;
}

int32_t Scale_32f_C1IR(float *imageData, int width, ImageRect &rect,
                       float ratio) {
  NppStatus status = NPP_ERROR;

  if (nullptr == imageData) {
    MBLOG_ERROR << "Parma is Null.";
    return static_cast<int32_t>(status);
  }

  if ((0 > width) || (PIXEL_THRESHOLD < width)) {
    MBLOG_ERROR << "image width is invalid.";
    return static_cast<int32_t>(status);
  }

  if (!CheckRoiValid(rect)) {
    return static_cast<int32_t>(status);
  }

  float *startPos = imageData + rect.y * width + rect.x;

  NppiSize oSizeROI;
  oSizeROI.height = rect.height;
  oSizeROI.width = rect.width;

  status = nppiMulC_32f_C1IR(ratio, (Npp32f *)startPos, width * sizeof(float),
                             oSizeROI);
  if (NPP_SUCCESS != status) {
    MBLOG_ERROR << "Scale_32f_C1R. Fail to scale. ratio:" << ratio;
  }

  return static_cast<int32_t>(status);
}