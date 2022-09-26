
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

#include "modelbox_jni.h"

#include "com_modelbox_ModelBox.h"
#include "modelbox/base/driver.h"
#include "modelbox/base/status.h"
#include "scoped_jvm.h"
#include "throw.h"

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_8) != JNI_OK) {
    return JNI_ERR;
  }

  modelbox::ScopedJvm::SetJavaVM(vm);

  return JNI_VERSION_1_8;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
  modelbox::ScopedJvm::SetJavaVM(nullptr);
}

/*
 * Class:     com_modelbox_ModelBox
 * Method:    SetDefaultScanPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBox_SetDefaultScanPath(
    JNIEnv *env, jclass j_class, jstring j_path) {
  const char *path = env->GetStringUTFChars(j_path, nullptr);
  if (path == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID, "invalid path");
    return;
  }

  modelbox::Drivers::SetDefaultScanPath(path);
  env->ReleaseStringUTFChars(j_path, path);
}

/*
 * Class:     com_modelbox_ModelBox
 * Method:    SetDefaultInfoPath
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBox_SetDefaultInfoPath(
    JNIEnv *env, jclass j_class, jstring j_path) {
  const char *path = env->GetStringUTFChars(j_path, nullptr);
  if (path == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID, "invalid path");
    return;
  }

  modelbox::Drivers::SetDefaultInfoPath(path);
  env->ReleaseStringUTFChars(j_path, path);
}