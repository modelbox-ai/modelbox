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

#include "java_module.h"

#include <jni.h>
#include <modelbox/base/log.h>

#include <chrono>
#include <functional>

std::shared_ptr<JavaJVM> kJavaJVM = nullptr;

JavaJVM::JavaJVM() {}

JavaJVM::~JavaJVM() {}

JNIEnv *JavaJVM::GetEnv() { return env_; }

modelbox::Status JavaJVM::InitJNI() {
  jsize vms_num = 0;
  auto ret = JNI_GetCreatedJavaVMs(nullptr, 0, &vms_num);
  if (vms_num <= 0) {
    JavaVMInitArgs vm_args;
    ret = JNI_CreateJavaVM(&jvm_, (void **)&env_, &vm_args);
    if (ret != JNI_OK) {
      return modelbox::STATUS_FAULT;
    }
    is_jvm_create_ = true;
  } else {
    JavaVM *jvms[vms_num];
    ret = JNI_GetCreatedJavaVMs(jvms, vms_num, &vms_num);
    if (ret != JNI_OK) {
      return modelbox::STATUS_FAULT;
    }
    jvm_ = jvms[0];
    jvm_->GetEnv((void **)&env_, JNI_VERSION_1_8);
  }

  is_initialized_ = true;
  return modelbox::STATUS_OK;
}

modelbox::Status JavaJVM::InitJVM() {
  auto ret = InitJNI();
  if (!ret) {
    return ret;
  }

  return modelbox::STATUS_OK;
}

modelbox::Status JavaJVM::ExitJNI() {
  if (is_initialized_ == false) {
    return modelbox::STATUS_OK;
  }

  if (is_jvm_create_ == false) {
    return modelbox::STATUS_OK;
  }

  jvm_->DestroyJavaVM();
  is_jvm_create_ = false;
  is_initialized_ = false;

  return modelbox::STATUS_OK;
}

modelbox::Status JavaJVM::ExitJVM() {
  ExitJNI();

  return modelbox::STATUS_OK;
}
