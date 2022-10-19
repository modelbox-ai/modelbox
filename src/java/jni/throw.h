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

#ifndef MODELBOX_JNI_THROW_H_
#define MODELBOX_JNI_THROW_H_

#include <jni.h>

#include "modelbox/base/status.h"

namespace modelbox {

constexpr const char *JNIEXCEPT_NullPointer = "java/lang/NullPointerException";
constexpr const char *JNIEXCEPT_OutOfMemoryError = "java/lang/OutOfMemoryError";
constexpr const char *JNIEXCEPT_RuntimeException = "java/lang/RuntimeException";
constexpr const char *JNIEXCEPT_IllegalArgumentException =
    "java/lang/IllegalArgumentException";

void ModelBoxJNIThrow(JNIEnv *env, StatusCode code,
                      const std::string &errormsg);

void ModelBoxJNIThrow(JNIEnv *env, Status &status);

void ModelBoxJNIThrow(JNIEnv *env, const char *runtime_exception,
                      const char *errmsg);

std::shared_ptr<Status> ModelboxJNICatchException(JNIEnv *env);

std::string ModelboxExceptionMsg(JNIEnv *env, std::string *stack = nullptr);

}  // namespace modelbox

#endif  // MODELBOX_JNI_THROW_H_
