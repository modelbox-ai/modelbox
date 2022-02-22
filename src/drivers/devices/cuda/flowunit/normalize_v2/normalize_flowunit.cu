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


#include "normalize_flowunit_cu.h"

__global__ void NormalizeAndCHWKernel(const uint8_t *input, int H, int W, int C,
                                      const float *normalize_mean,
                                      const float *normalize_std,
                                      float *output) {
  const int n = blockIdx.x;
  const int stride = H * W * C;

  const uint8_t *in = input + n * stride;
  float *out = output + n * stride;
  int c = 0;
  int h = 0;
  int w = 0;

  for (c = 0; c < C; ++c) {
    for (h = threadIdx.y; h < H; h += blockDim.y) {
      for (w = threadIdx.x; w < W; w += blockDim.x) {
        out[w + c * H * W + h * W] =
            static_cast<float>((static_cast<float>(in[c + h * W * C + w * C]) -
                                normalize_mean[c]) *
                               normalize_std[c]);
      }
    }
  }
}

__global__ void NormalizeKernel(const uint8_t *input, int H, int W, int C,
                                const float *normalize_mean,
                                const float *normalize_std, float *output) {
  const int n = blockIdx.x;
  const int stride = H * W * C;

  const uint8_t *in = input + n * stride;
  float *out = output + n * stride;

  int c = 0;
  int h = 0;
  int w = 0;

  for (c = 0; c < C; ++c) {
    for (h = threadIdx.y; h < H; h += blockDim.y) {
      for (w = threadIdx.x; w < W; w += blockDim.x) {
        out[c + h * W * C + w * C] =
            static_cast<float>((static_cast<float>(in[c + h * W * C + w * C]) -
                                normalize_mean[c]) *
                               normalize_std[c]);
      }
    }
  }
}

void NormalizeAndCHW(const uint8_t *input, int N, int H, int W, int C,
                     const float *normalize_mean, const float *normalize_std,
                     float *output, cudaStream_t stream) {
  constexpr int BLOCK_X = 32;
  constexpr int BLOCK_Y = 32;

  NormalizeAndCHWKernel<<<N, dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
      input, H, W, C, normalize_mean, normalize_std, output);
  return;
}

void Normalize(const uint8_t *input, int N, int H, int W, int C,
               const float *normalize_mean, const float *normalize_std,
               float *output, cudaStream_t stream) {
  constexpr int BLOCK_X = 32;
  constexpr int BLOCK_Y = 32;

  NormalizeKernel<<<N, dim3(BLOCK_X, BLOCK_Y), 0, stream>>>(
      input, H, W, C, normalize_mean, normalize_std, output);
  return;
}