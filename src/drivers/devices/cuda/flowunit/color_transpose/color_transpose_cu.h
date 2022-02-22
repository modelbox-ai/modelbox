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


#include <npp.h>
#include <stdint.h>
#include "cuda_runtime.h"

constexpr int COLOR_CHANNEL_COUNT = 3;
constexpr int GRAY_CHANNEL_COUNT = 1;

NppStatus RGBToBGR(NppiSize &size, const uint8_t *input_data,
                   uint8_t *output_data, cudaStream_t stream);

NppStatus BGRToRGB(NppiSize &size, const uint8_t *input_data,
                   uint8_t *output_data, cudaStream_t stream);

NppStatus GRAYToRGB(NppiSize &size, const uint8_t *input_data,
                    uint8_t *output_data, cudaStream_t stream);

NppStatus GRAYToBGR(NppiSize &size, const uint8_t *input_data,
                    uint8_t *output_data, cudaStream_t stream);