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


#include <modelbox/base/base64_simd.h>

#include <cpuid.h>
#include <immintrin.h>
#include "modelbox/base/log.h"

namespace modelbox {

struct base64_encode_simd_param {
  const int group_size{3};
  const int result_group_size{4};
  const int batch_size{12 * 4};
  const __m512i idx0_1 =
      _mm512_setr_epi32(0x0, 0x1, 0x2, 0xC, 0x3, 0x4, 0x5, 0xD, 0x6, 0x7, 0x8,
                        0xE, 0x9, 0xA, 0xB, 0xF);
  const __m512i idx1_2 = _mm512_setr_epi32(
      0x01020001, 0x04050304, 0x07080607, 0x0A0B090A, 0x01020001, 0x04050304,
      0x07080607, 0x0A0B090A, 0x01020001, 0x04050304, 0x07080607, 0x0A0B090A,
      0x01020001, 0x04050304, 0x07080607, 0x0A0B090A);
  const char *encode_table =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int batch_count{0};
  int group_count{0};
  int input_len{0};
  int output_len{0};
};

// NOLINTNEXTLINE
void cpuid(int info[4], int function_id_) {
  __cpuid_count(function_id_, 0, info[0], info[1], info[2], info[3]);
}

__m512i ConvertAscii(__m512i &index) {
  // index:0 ~ 25 -> 'A'(65) - 'X',  offset: +65
  auto mask_upper_letter = _mm512_cmple_epi8_mask(index, _mm512_set1_epi8(25));
  auto result =
      _mm512_maskz_add_epi8(mask_upper_letter, index, _mm512_set1_epi8(65));

  // index:26 ~ 51 -> 'a'(97) - 'z',  offset: +71
  auto mask_lower_letter_0 =
      _mm512_cmpge_epi8_mask(index, _mm512_set1_epi8(26));
  auto mask_lower_letter_1 =
      _mm512_cmple_epi8_mask(index, _mm512_set1_epi8(51));
  result =
      _mm512_mask_add_epi8(result, mask_lower_letter_0 & mask_lower_letter_1,
                           index, _mm512_set1_epi8(71));

  // index:52 ~ 61 -> '0'(48) ~ '9' , offset: -4
  auto mask_num_0 = _mm512_cmpge_epi8_mask(index, _mm512_set1_epi8(52));
  auto mask_num_1 = _mm512_cmple_epi8_mask(index, _mm512_set1_epi8(61));
  result = _mm512_mask_sub_epi8(result, mask_num_0 & mask_num_1, index,
                                _mm512_set1_epi8(4));

  // index:62 -> '+'(43), offset: -19
  auto mask_plus = _mm512_cmpeq_epi8_mask(index, _mm512_set1_epi8(62));
  result = _mm512_mask_sub_epi8(result, mask_plus, index, _mm512_set1_epi8(19));

  // index:63 -> '/'(47), offset: -16
  auto mask_slash = _mm512_cmpeq_epi8_mask(index, _mm512_set1_epi8(63));
  result =
      _mm512_mask_sub_epi8(result, mask_slash, index, _mm512_set1_epi8(16));

  return result;
}

void Base64EncodeLessOneBatch(struct base64_encode_simd_param param,
                              const uint8_t *src, uint8_t *output_buffer) {
  // less than one batch
  int i = param.batch_size * param.batch_count;
  while (i + param.group_size <= param.input_len) {
    output_buffer[0] = param.encode_table[src[i] >> 2];
    output_buffer[1] =
        param.encode_table[((src[i] << 4) | (src[i + 1] >> 4)) & 0x3F];
    output_buffer[2] =
        param.encode_table[((src[i + 1] << 2) | (src[i + 2] >> 6)) & 0x3F];
    output_buffer[3] = param.encode_table[src[i + 2] & 0x3F];
    output_buffer += param.result_group_size;
    i += param.group_size;
  }

  // Less than one group
  int remind_byte = param.input_len % 3;
  if (remind_byte == 1) {
    output_buffer[0] = param.encode_table[(src[i] & 0xFC) >> 2];
    output_buffer[1] = param.encode_table[((src[i] & 0x03) << 4)];
    output_buffer[2] = '=';
    output_buffer[3] = '=';
  } else if (remind_byte == 2) {
    output_buffer[0] = param.encode_table[(src[i] & 0xFC) >> 2];
    output_buffer[1] =
        param.encode_table[((src[i] & 0x03) << 4) | ((src[i + 1] & 0xF0) >> 4)];
    output_buffer[2] = param.encode_table[((src[i + 1] & 0x0F) << 2)];
    output_buffer[3] = '=';
  }
}

bool CheckSupportBase64SIMD() {
  // check cpu whether support avx512f and avx512bw
  int info[4] = {0};
  const int function_id = 0x00000007;
  cpuid(info, function_id);
  bool HAS_AVX512F = (info[1] & ((int)1 << 16)) != 0;
  bool HAS_AVX512BW = (info[1] & ((int)1 << 30)) != 0;

  return HAS_AVX512F && HAS_AVX512BW;
}

Status Base64EncodeSIMD(const uint8_t *input, size_t input_len,
                        std::string *output) {
  if (input == nullptr || input_len == 0) {
    const auto *err_msg = "base64 encode input data is null or size is zero";
    MBLOG_ERROR << err_msg;
    return {STATUS_INVALID, err_msg};
  }

  if (!CheckSupportBase64SIMD()) {
    return {STATUS_NOTFOUND, "not support simd."};
  }

  struct base64_encode_simd_param param;
  param.input_len = input_len;
  param.group_count = (input_len + 2) / 3;
  param.output_len = param.group_count * 4;
  param.batch_count = input_len / param.batch_size;

  output->resize(param.output_len);
  auto *output_buffer = (uint8_t *)output->data();

  for (int i = 0; i <= param.input_len - param.batch_size;
       i += param.batch_size) {
    // get index
    // in0 =
    // [XXXX|XXXX|XXXX|XXXX|PPPO|OONN|NMMM|LLLK|KKJJ|JIII|HHHG|GGFF|FEEE|DDDC|CCBB|BAAA]
    __m512i in0 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));

    // in1 =
    // [XXXX|PPPO|OONN|NMMM|XXXX|LLLK|KKJJ|JIII|XXXX|HHHG|GGFF|FEEE|XXXX|DDDC|CCBB|BAAA]
    __m512i in1 = _mm512_permutexvar_epi32(param.idx0_1, in0);

    // [XAAA] -> [xxxxxxxx|ccdddddd|bbbbcccc|aaaaaabb] ([3|2|1|0])
    //        -> [bbbbcccc|ccdddddd|aaaaaabb|bbbbcccc] ([1|2|0|1])
    // in2 = [...|D1D2D0D1|C1C2C0C1|B1B2B0B1|A1A2A0A1]
    __m512i in2 = _mm512_shuffle_epi8(in1, param.idx1_2);

    //    [bbbbcccc|ccdddddd|aaaaaabb|bbbbcccc]
    // -> [00dddddd|00cccccc|00bbbbbb|00aaaaaa]
    //      byte3    byte2    byte1    byte0

    // byte0 & byte2
    //    [bbbbcccc|ccdddddd|aaaaaabb|bbbbcccc]
    // -> [0000cccc|cc000000|aaaaaa00|00000000]
    // -> [00000000|00cccccc|00000000|00aaaaaa]
    __m512i byte_0_2 = _mm512_and_si512(in2, _mm512_set1_epi32(0x0fc0fc00));
    byte_0_2 = _mm512_srlv_epi16(byte_0_2, _mm512_set1_epi32(0x0006000a));

    // byte1 & byte3
    //    [bbbbcccc|ccdddddd|aaaaaabb|bbbbcccc]
    // -> [00000000|00dddddd|000000bb|bbbb0000]
    // -> [00dddddd|00000000|00bbbbbb|00000000]
    __m512i byte_1_3 = _mm512_and_si512(in2, _mm512_set1_epi32(0x003f03f0));
    byte_1_3 = _mm512_sllv_epi16(byte_1_3, _mm512_set1_epi32(0x00080004));
    __m512i index = _mm512_or_epi32(byte_1_3, byte_0_2);

    // convert to ascii
    auto result = ConvertAscii(index);

    // save result
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output_buffer), result);
    output_buffer += 64;
  }

  // less than one batch
  Base64EncodeLessOneBatch(param, input, output_buffer);
  return STATUS_OK;
}
}  // namespace modelbox