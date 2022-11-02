
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

#include "jni_native_object.h"

#include "modelbox/base/log.h"
#include "modelbox/base/status.h"

namespace modelbox {

JNINativeObject::JNINativeObject(jobject jni_object,
                                 std::shared_ptr<void> native_shared_ptr)
    : jni_object_(jni_object),
      native_shared_ptr_(std::move(native_shared_ptr)){};

JNINativeObject::JNINativeObject() = default;

JNINativeObject::~JNINativeObject() = default;

jobject JNINativeObject::GetJObject() { return jni_object_; }

void JNINativeObject::SetJObject(jobject object) { jni_object_ = object; }

std::shared_ptr<void> JNINativeObject::GetNativeSharedPtr() {
  return native_shared_ptr_;
}

Status JNINativeObject::SetNativeSharedPtr(
    JNIEnv *env, jlong handle, const std::shared_ptr<void> &native_shared_ptr) {
  Status ret = STATUS_SUCCESS;
  auto *native_object = FromHandle(handle);
  if (native_object == nullptr) {
    return {STATUS_INVALID, "handle is invalid"};
  }

  native_object->SetNativeSharedPtr(native_shared_ptr);
  return ret;
}

Status JNINativeObject::SetNativeSharedPtr(
    JNIEnv *env, jobject object, const std::shared_ptr<void> &native_shared_ptr,
    const char *member) {
  Status ret = STATUS_SUCCESS;
  auto *native_object = FromJObject(env, object, member);
  if (native_object == nullptr) {
    return modelbox::StatusError;
  }

  native_object->SetNativeSharedPtr(native_shared_ptr);
  return ret;
}

void JNINativeObject::SetNativeSharedPtr(
    std::shared_ptr<void> native_shared_ptr) {
  native_shared_ptr_ = std::move(native_shared_ptr);
}

JNINativeObject *JNINativeObject::FromHandle(jlong handle) {
  return reinterpret_cast<JNINativeObject *>(handle);
}

JNINativeObject *JNINativeObject::FromJObject(JNIEnv *env, jobject object,
                                              const char *member) {
  if (env == nullptr || object == nullptr) {
    std::string errmsg = "get jni native from object failed, invalid argument";
    StatusError = {STATUS_INVALID, errmsg};
    return nullptr;
  }

  jclass cls = env->GetObjectClass(object);
  if (cls == nullptr) {
    std::string errmsg = "get object class failed.";
    StatusError = {STATUS_INVALID, errmsg};
    return nullptr;
  }
  Defer { env->DeleteLocalRef(cls); };

  jfieldID ptrField = env->GetFieldID(cls, member, "J");
  if (ptrField == nullptr) {
    std::string errmsg = "not a modelbox object, not extends from NativeObject";
    StatusError = {STATUS_INVALID, errmsg};
    return nullptr;
  }

  auto handle = (jlong)env->GetLongField(object, ptrField);
  if (handle == 0) {
    std::string errmsg = "native handler is invalid";
    StatusError = {STATUS_INVALID, errmsg};
    return nullptr;
  }

  auto *native_object = FromHandle(handle);
  if (native_object == nullptr) {
    return nullptr;
  }

  return native_object;
}

jlong JNINativeObject::NewHandle(
    jobject object, const std::shared_ptr<void> &native_shared_ptr) {
  auto *native_object = new JNINativeObject(object, native_shared_ptr);
  return (jlong)native_object;
}

jobject JNINativeObject::NewJObject(
    JNIEnv *env, const char *clazz,
    const std::shared_ptr<void> &native_shared_ptr, const char *member) {
  jclass cls = env->FindClass(clazz);
  if (cls == nullptr) {
    StatusError = {STATUS_INVALID, "cannot find class " + std::string(clazz)};
    return nullptr;
  }
  Defer { env->DeleteLocalRef(cls); };

  auto *cls_constructor = env->GetMethodID(cls, "<init>", "()V");
  if (cls_constructor == nullptr) {
    std::string errmsg =
        "cannot find constructor for " + std::string(clazz) + ".";
    modelbox::StatusError = {modelbox::STATUS_INVALID, errmsg};
  }

  jobject object = env->NewObject(cls, cls_constructor);
  if (object == nullptr) {
    std::string errmsg = "new object for " + std::string(clazz) + " failed.";
    modelbox::StatusError = {modelbox::STATUS_NOMEM, errmsg};
    return nullptr;
  }

  jfieldID ptrField = env->GetFieldID(cls, member, "J");
  if (ptrField == nullptr) {
    std::string errmsg =
        "not a modelbox class, not extends from NativeObject, ";
    errmsg += "class: " + std::string(clazz);
    modelbox::StatusError = {modelbox::STATUS_INVALID, errmsg};
    env->DeleteLocalRef(object);
    return nullptr;
  }

  jlong oldptr = env->GetLongField(object, ptrField);
  if (oldptr != 0) {
    DeleteHandle(oldptr);
  }

  auto native_object = NewHandle(object, native_shared_ptr);
  env->SetLongField(object, ptrField, native_object);
  return object;
}

void JNINativeObject::DeleteHandle(jlong handle) {
  auto *native_object = FromHandle(handle);
  if (native_object == nullptr) {
    return;
  }

  delete native_object;
}

void JNINativeObject::DeleteJObject(JNIEnv *env, jobject object,
                                    const char *member) {
  auto *native_object = FromJObject(env, object, member);
  if (native_object == nullptr) {
    return;
  }
  
  jclass cls = env->GetObjectClass(object);
  if (cls == nullptr) {
    std::string errmsg = "get object class failed.";
    StatusError = {STATUS_INVALID, errmsg};
    return;
  }
  Defer { env->DeleteLocalRef(cls); };

  jfieldID ptrField = env->GetFieldID(cls, member, "J");
  if (ptrField == nullptr) {
    std::string errmsg = "not a modelbox object, not extends from NativeObject";
    StatusError = {STATUS_INVALID, errmsg};
    return;
  }
  env->SetLongField(object, ptrField, 0);

  delete native_object;
}

}  // namespace modelbox