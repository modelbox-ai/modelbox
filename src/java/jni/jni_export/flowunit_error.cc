
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

#include <memory>

#include "com_modelbox_FlowUnitError.h"
#include "jni_native_object.h"
#include "modelbox/error.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_FlowUnitError
 * Method:    FlowUnitError_New
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_com_modelbox_FlowUnitError_FlowUnitError_1New__Ljava_lang_String_2(
    JNIEnv *env, jobject j_this, jstring j_desc) {
  if (j_desc == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::JNIEXCEPT_NullPointer,
                               "input argument is null");
    return 0;
  }

  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::FlowUnitError>(
                  modelbox::jstring2string(env, j_desc)));
}

/*
 * Class:     com_modelbox_FlowUnitError
 * Method:    FlowUnitError_New
 * Signature: (Ljava/lang/String;Ljava/lang/String;Lcom/modelbox/Status;)J
 */
JNIEXPORT jlong JNICALL
Java_com_modelbox_FlowUnitError_FlowUnitError_1New__Ljava_lang_String_2Ljava_lang_String_2Lcom_modelbox_Status_2(
    JNIEnv *env, jobject j_this, jstring j_node, jstring j_pos,
    jobject j_status) {
  if (j_node == nullptr || j_pos == nullptr || j_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::JNIEXCEPT_NullPointer,
                               "input argument is null");
    return 0;
  }

  auto n_flowunit_status =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(env,
                                                                      j_status);
  if (j_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::FlowUnitError>(
                  modelbox::jstring2string(env, j_node),
                  modelbox::jstring2string(env, j_pos), *n_flowunit_status));
}

/*
 * Class:     com_modelbox_FlowUnitError
 * Method:    FlowUnitError_GetDesc
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_FlowUnitError_FlowUnitError_1GetDesc(JNIEnv *env,
                                                       jobject j_this) {
  auto n_flowunit_error =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitError>(
          env, j_this);
  if (n_flowunit_error == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_flowunit_error->GetDesc().c_str());
}

/*
 * Class:     com_modelbox_FlowUnitError
 * Method:    FlowUnitError_GetStatus
 * Signature: ()Lcom/modelbox/Status;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_FlowUnitError_FlowUnitError_1GetStatus(JNIEnv *env,
                                                         jobject j_this) {
  auto n_flowunit_error =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitError>(
          env, j_this);
  if (n_flowunit_error == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto stat = std::make_shared<modelbox::Status>(n_flowunit_error->GetStatus());

  auto *j_status =
      modelbox::JNINativeObject::NewJObject(env, "com/modelbox/Status", stat);
  modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
  return j_status;
}
