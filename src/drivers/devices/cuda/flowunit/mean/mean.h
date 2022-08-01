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


#ifndef MODELBOX_MEAN_H_
#define MODELBOX_MEAN_H_

#include <cuda_runtime.h>
#include <nppdefs.h>

#include <iostream>

#include "modelbox/base/log.h"

#define PIXEL_THRESHOLD 4096

/**
 * 2D Rectangle
 * This struct contains position and size information of a rectangle in
 * two space.
 * The rectangle's position is usually signified by the coordinate of its
 * upper-left corner.
 */
typedef struct {
  int x;     /**<  x-coordinate of upper left corner (lowest memory address). */
  int y;     /**<  y-coordinate of upper left corner (lowest memory address). */
  int width; /**<  Rectangle width. */
  int height; /**<  Rectangle height. */
} ImageRect;

typedef struct {
  float channel_0;
  float channel_1;
  float channel_2;
} ImageMean_32f;

bool CheckRoiValid(const ImageRect &roi);

int32_t Mean_PLANAR_32f_P3R(const float *pSrcPlanarData, int width, int height,
                            const ImageRect &srcRoi, const ImageMean_32f &mean,
                            cudaStream_t stream);

#endif  // MODELBOX_MEAN_H_
