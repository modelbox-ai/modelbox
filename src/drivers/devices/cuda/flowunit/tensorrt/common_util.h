
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

#ifndef MODELBOX_TENSORRT_COMMOM_H_
#define MODELBOX_TENSORRT_COMMOM_H_

#include <NvInfer.h>

#include <memory>

inline int Volume(nvinfer1::Dims dims) {
  auto* begin_dim_d = (dims.d[0] == -1 ? (dims.d + 1) : dims.d);
  return std::accumulate(begin_dim_d, dims.d + dims.nbDims, 1,
                         std::multiplies<int>());
}

struct TensorRTInferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj == nullptr) {
      return;
    }

#ifdef TENSORRT8
    delete obj;
#else
    obj->destroy();
#endif
  }
};

template <typename T>
inline std::shared_ptr<T> TensorRTInferObject(T* obj) {
  if (obj == nullptr) {
    return nullptr;
  }
  
  return std::shared_ptr<T>(obj, TensorRTInferDeleter());
}

#endif