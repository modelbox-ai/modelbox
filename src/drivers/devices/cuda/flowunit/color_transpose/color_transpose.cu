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


#include "color_transpose_cu.h"

__global__ void TransposeRGBToBGR(const uint8_t *rgb_input, uint8_t *bgr_output,
                                  unsigned int images_size) {
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= images_size) {
    return;
  }

  const uint8_t *in = &rgb_input[idx * COLOR_CHANNEL_COUNT];
  uint8_t *out = &bgr_output[idx * COLOR_CHANNEL_COUNT];

  out[0] = in[2];
  out[1] = in[1];
  out[2] = in[0];
}

__global__ void TransposeGrayToRGB(const uint8_t *gray_input,
                                   uint8_t *rgb_output,
                                   unsigned int images_size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= images_size) {
    return;
  }

  const uint8_t in = gray_input[idx];
  uint8_t *out = &rgb_output[idx * COLOR_CHANNEL_COUNT];

  out[0] = in;
  out[1] = in;
  out[2] = in;
}

auto TransposeBGRToRGB = TransposeRGBToBGR;

constexpr int MAX_CUDA_BLOCK_THD_SIZE = 1024;

NppStatus RGBToBGR(NppiSize &size, const uint8_t *input_data,
                   uint8_t *output_data, cudaStream_t stream) {
  // For CUDA kernel
  const unsigned int total_size = size.height * size.width;
  const unsigned int block = total_size < MAX_CUDA_BLOCK_THD_SIZE
                                 ? total_size
                                 : MAX_CUDA_BLOCK_THD_SIZE;
  const unsigned int grid = (total_size + block - 1) / block;

  // RGB -> BGR
  TransposeRGBToBGR<<<grid, block, 0, stream>>>(input_data, output_data,
                                                total_size);
  return NPP_SUCCESS;
}

NppStatus BGRToRGB(NppiSize &size, const uint8_t *input_data,
                   uint8_t *output_data, cudaStream_t stream) {
  // For CUDA kernel
  const unsigned int total_size = size.height * size.width;
  const unsigned int block = total_size < MAX_CUDA_BLOCK_THD_SIZE
                                 ? total_size
                                 : MAX_CUDA_BLOCK_THD_SIZE;
  const unsigned int grid = (total_size + block - 1) / block;

  // BGR -> RGB
  TransposeBGRToRGB<<<grid, block, 0, stream>>>(input_data, output_data,
                                                total_size);
  return NPP_SUCCESS;
}

NppStatus GRAYToRGB(NppiSize &size, const uint8_t *input_data,
                    uint8_t *output_data, cudaStream_t stream) {
  const unsigned int total_size = size.height * size.width;
  const unsigned int block = total_size < MAX_CUDA_BLOCK_THD_SIZE
                                 ? total_size
                                 : MAX_CUDA_BLOCK_THD_SIZE;
  const unsigned int grid = (total_size + block - 1) / block;

  TransposeGrayToRGB<<<grid, block, 0, stream>>>(input_data, output_data,
                                                 total_size);
  return NPP_SUCCESS;
}

NppStatus GRAYToBGR(NppiSize &size, const uint8_t *input_data,
                    uint8_t *output_data, cudaStream_t stream) {
  const unsigned int total_size = size.height * size.width;
  const unsigned int block = total_size < MAX_CUDA_BLOCK_THD_SIZE
                                 ? total_size
                                 : MAX_CUDA_BLOCK_THD_SIZE;
  const unsigned int grid = (total_size + block - 1) / block;

  TransposeGrayToRGB<<<grid, block, 0, stream>>>(input_data, output_data,
                                                 total_size);
  return NPP_SUCCESS;
}
