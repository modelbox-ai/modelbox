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

#ifndef MODELBOX_JNI_BIND_H_
#define MODELBOX_JNI_BIND_H_

#include <jni.h>

#include <memory>
#include <string>

#include "modelbox/base/status.h"

namespace modelbox {

constexpr const char *NATIVE_HANDLER_MEMBER_NAME = "native_handle";

class JNINativeObject {
 public:
  JNINativeObject(jobject jni_object, std::shared_ptr<void> native_shared_ptr);
  virtual ~JNINativeObject();

  jobject GetJObject();

  void SetJObject(jobject object);

  std::shared_ptr<void> GetNativeSharedPtr();

  template <typename T>
  inline std::shared_ptr<T> GetNativeSharedPtr() {
    return std::static_pointer_cast<T>(GetNativeSharedPtr());
  }

  static jlong NewHandle(jobject object,
                         const std::shared_ptr<void> &native_shared_ptr);

  static void DeleteHandle(jlong handle);

  static jobject NewJObject(JNIEnv *env, const char *clazz,
                            const std::shared_ptr<void> &native_shared_ptr,
                            const char *member = NATIVE_HANDLER_MEMBER_NAME);

  static void DeleteJObject(JNIEnv *env, jobject object,
                            const char *member = NATIVE_HANDLER_MEMBER_NAME);

  template <typename T>
  inline static std::shared_ptr<T> GetNativeSharedPtr(jlong handle) {
    auto native_object = FromHandle(handle);
    if (native_object == nullptr) {
      return nullptr;
    }

    return native_object->GetNativeSharedPtr<T>();
  }

  template <typename T>
  inline static std::shared_ptr<T> GetNativeSharedPtr(
      JNIEnv *env, jobject object,
      const char *member = NATIVE_HANDLER_MEMBER_NAME) {
    auto native_object = FromJObject(env, object, member);
    if (native_object == nullptr) {
      return nullptr;
    }

    return native_object->GetNativeSharedPtr<T>();
  }

  static Status SetNativeSharedPtr(
      JNIEnv *env, jlong handle,
      const std::shared_ptr<void> &native_shared_ptr);

  static Status SetNativeSharedPtr(
      JNIEnv *env, jobject object,
      const std::shared_ptr<void> &native_shared_ptr,
      const char *member = NATIVE_HANDLER_MEMBER_NAME);

 private:
  JNINativeObject() = default;
  void SetNativeSharedPtr(std::shared_ptr<void> native_shared_ptr);

  static JNINativeObject *FromHandle(jlong handle);

  static JNINativeObject *FromJObject(
      JNIEnv *env, jobject object,
      const char *member = NATIVE_HANDLER_MEMBER_NAME);

  jobject jni_object_;
  std::shared_ptr<void> native_shared_ptr_;
};

}  // namespace modelbox

#endif  // MODELBOX_JNI_BIND_H_
