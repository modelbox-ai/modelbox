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

#ifndef IMAGE_ROTATE_CU_H_
#define IMAGE_ROTATE_CU_H_

#include <stdint.h>

#include "cuda_runtime.h"

int32_t ClockWiseRotateGPU(const u_char *srcData, u_char *dstData,
                           int32_t height, int32_t width, int32_t rotateAngle,
                           cudaStream_t stream);

__global__ void RotateImg_u8c3r(const u_char *srcData, u_char *dstData,
                                int32_t width, int32_t height,
                                int32_t rotateAngle);

__global__ void RotateImg_u8p3(const u_char *srcData, u_char *dstData,
                               int width, int height, int32_t rotateAngle);

#endif