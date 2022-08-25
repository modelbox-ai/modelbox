
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

#include "scoped_jvm.h"

#include <memory>

#include "modelbox/base/log.h"

namespace modelbox {

JavaVM *ScopedJvm::jvm_;

JavaVM *ScopedJvm::GetJavaVM() { return jvm_; }

void ScopedJvm::SetJavaVM(JavaVM *vm) { jvm_ = vm; }

ScopedJvm::ScopedJvm() {
  JNIEnv *env = nullptr;
  if (jvm_ == nullptr) {
    throw std::runtime_error("jvm pointer is not set");
    return;
  }

  auto ret = jvm_->GetEnv((void **)&env, JNI_VERSION_1_6);
  if (ret == JNI_OK) {
    env_ = env;
    return;
  }

  ret = jvm_->AttachCurrentThread((void **)&env, nullptr);
  if (ret != JNI_OK) {
    throw std::runtime_error("Attach jvm thread failed.");
    return;
  }

  env_ = env;
  do_attach_ = true;
}

ScopedJvm::~ScopedJvm() {
  if (jvm_ == nullptr) {
    return;
  }

  if (do_attach_ == true) {
    jvm_->DetachCurrentThread();
  }

  env_ = nullptr;
}

JNIEnv *ScopedJvm::GetJNIEnv() { return env_; }

}  // namespace modelbox