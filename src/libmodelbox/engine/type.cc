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


#include <modelbox/type.h>

namespace modelbox {

size_t GetDataTypeSize(ModelBoxDataType type) {
#define CASE(T) \
  case T:       \
    return DataTypeSize<T>::Size;
  switch (type) {
    CASE(MODELBOX_FLOAT);
    CASE(MODELBOX_DOUBLE);
    CASE(MODELBOX_INT32);
    CASE(MODELBOX_UINT32);
    CASE(MODELBOX_UINT16);
    CASE(MODELBOX_UINT8);
    CASE(MODELBOX_INT16);
    CASE(MODELBOX_INT8);
    CASE(MODELBOX_STRING);
    CASE(MODELBOX_INT64);
    CASE(MODELBOX_UINT64);
    CASE(MODELBOX_BOOL);
    CASE(MODELBOX_HALF);
    default:
      return 0;
  }
#undef CASE
}
}  // namespace modelbox