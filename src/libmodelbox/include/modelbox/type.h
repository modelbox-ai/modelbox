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


#ifndef MODELBOX_TYPE_H_
#define MODELBOX_TYPE_H_

#include <modelbox/base/log.h>
#include <stdint.h>

namespace modelbox {

typedef enum ModelBoxDataType {
  MODELBOX_TYPE_INVALID = 0,
  MODELBOX_FLOAT = 1,
  MODELBOX_DOUBLE = 2,
  MODELBOX_INT32 = 3,  // Int32 tensors are always in 'host' memory.
  MODELBOX_UINT8 = 4,
  MODELBOX_INT16 = 5,
  MODELBOX_INT8 = 6,
  MODELBOX_STRING = 7,
  MODELBOX_COMPLEX64 = 8,  // Single-precision complex
  MODELBOX_COMPLEX = 8,    // Old identifier kept for API backwards compatibility
  MODELBOX_INT64 = 9,
  MODELBOX_BOOL = 10,
  MODELBOX_QINT8 = 11,     // Quantized int8
  MODELBOX_QUINT8 = 12,    // Quantized uint8
  MODELBOX_QINT32 = 13,    // Quantized int32
  MODELBOX_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  MODELBOX_QINT16 = 15,    // Quantized int16
  MODELBOX_QUINT16 = 16,   // Quantized uint16
  MODELBOX_UINT16 = 17,
  MODELBOX_COMPLEX128 = 18,  // Double-precision complex
  MODELBOX_HALF = 19,
  MODELBOX_RESOURCE = 20,
  MODELBOX_VARIANT = 21,
  MODELBOX_UINT32 = 22,
  MODELBOX_UINT64 = 23,
} ModelBoxDataType;

template <class T>
struct TypeToDataType;

template <ModelBoxDataType T>
struct DataTypeSize;

template <ModelBoxDataType T>
struct DataTypeToType;

struct Float16 {
  uint8_t bytes[2];
};

#define MODELBOX_DATATYPE_DEFINE(TYPE, DATA_TYPE)                 \
  template <>                                          \
  struct TypeToDataType<TYPE> {                        \
    static constexpr ModelBoxDataType Value = DATA_TYPE; \
  };                                                   \
  template <>                                          \
  struct DataTypeSize<DATA_TYPE> {                     \
    static constexpr size_t Size = sizeof(TYPE);       \
  };                                                   \
  template <>                                          \
  struct DataTypeToType<DATA_TYPE> {                   \
    typedef TYPE Type;                                 \
  }

MODELBOX_DATATYPE_DEFINE(float, MODELBOX_FLOAT);
MODELBOX_DATATYPE_DEFINE(double, MODELBOX_DOUBLE);
MODELBOX_DATATYPE_DEFINE(int32_t, MODELBOX_INT32);
MODELBOX_DATATYPE_DEFINE(uint32_t, MODELBOX_UINT32);
MODELBOX_DATATYPE_DEFINE(uint16_t, MODELBOX_UINT16);
MODELBOX_DATATYPE_DEFINE(uint8_t, MODELBOX_UINT8);
MODELBOX_DATATYPE_DEFINE(int16_t, MODELBOX_INT16);
MODELBOX_DATATYPE_DEFINE(int8_t, MODELBOX_INT8);
MODELBOX_DATATYPE_DEFINE(std::string, MODELBOX_STRING);
MODELBOX_DATATYPE_DEFINE(int64_t, MODELBOX_INT64);
MODELBOX_DATATYPE_DEFINE(uint64_t, MODELBOX_UINT64);
MODELBOX_DATATYPE_DEFINE(bool, MODELBOX_BOOL);
MODELBOX_DATATYPE_DEFINE(Float16, MODELBOX_HALF);

#undef MODELBOX_DATATYPE_DEFINE

extern size_t GetDataTypeSize(ModelBoxDataType type);

}  // namespace modelbox

#endif
