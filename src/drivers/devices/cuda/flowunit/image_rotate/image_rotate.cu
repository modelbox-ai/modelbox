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

#include "image_rotate_cu.h"
#include "math.h"

#define GPU_BLOCK_SIZE_X 16
#define GPU_BLOCK_SIZE_Y 16

#define GET_GRID_SIZE(gridSizeX, gridSizeY, height, width) \
  gridSizeX = ceil((width) / (GPU_BLOCK_SIZE_X * 1.0));    \
  gridSizeY = ceil((height) / (GPU_BLOCK_SIZE_Y * 1.0));

int32_t ClockWiseRotateGPU(const u_char *srcData, u_char *dstData,
                           int32_t height, int32_t width, int32_t rotateAngle,
                           cudaStream_t stream) {
  if (srcData == nullptr) {
    return -1;
  }

  cudaMemset(dstData, 0, height * width * 3 * sizeof(u_char));

  dim3 blockSize(GPU_BLOCK_SIZE_X, GPU_BLOCK_SIZE_Y);
  int32_t gridSizeX, gridSizeY;
  GET_GRID_SIZE(gridSizeX, gridSizeY, height, width);
  dim3 gridSize(gridSizeX, gridSizeY);

  RotateImg_u8c3r<<<gridSize, blockSize, 0, stream>>>(srcData, dstData, width,
                                                      height, rotateAngle);
  return 0;
}

__global__ void RotateImg_u8c3r(const u_char *srcData, u_char *dstData,
                                int32_t width, int32_t height,
                                int32_t rotateAngle) {
  const long tidX = blockIdx.x * blockDim.x + threadIdx.x;
  const long tidY = blockIdx.y * blockDim.y + threadIdx.y;

  if ((tidX >= width) || (tidY >= height)) {
    return;
  }

  if (rotateAngle == 90) {
    long rotateX = height - 1 - tidY;
    long rotateY = tidX;

    dstData[(rotateY * height + rotateX) * 3] =
        srcData[(tidY * width + tidX) * 3];
    dstData[(rotateY * height + rotateX) * 3 + 1] =
        srcData[(tidY * width + tidX) * 3 + 1];
    dstData[(rotateY * height + rotateX) * 3 + 2] =
        srcData[(tidY * width + tidX) * 3 + 2];
  } else if (rotateAngle == 180) {
    long rotateX = width - 1 - tidX;
    long rotateY = height - 1 - tidY;

    dstData[(rotateY * width + rotateX) * 3] =
        srcData[(tidY * width + tidX) * 3];
    dstData[(rotateY * width + rotateX) * 3 + 1] =
        srcData[(tidY * width + tidX) * 3 + 1];
    dstData[(rotateY * width + rotateX) * 3 + 2] =
        srcData[(tidY * width + tidX) * 3 + 2];
  } else if (rotateAngle == 270) {
    long rotateX = tidY;
    long rotateY = width - 1 - tidX;

    dstData[(rotateY * height + rotateX) * 3] =
        srcData[(tidY * width + tidX) * 3];
    dstData[(rotateY * height + rotateX) * 3 + 1] =
        srcData[(tidY * width + tidX) * 3 + 1];
    dstData[(rotateY * height + rotateX) * 3 + 2] =
        srcData[(tidY * width + tidX) * 3 + 2];
  }
}
