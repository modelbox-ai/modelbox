
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

#include "modelbox/session_context.h"

#include <memory>

#include "com_modelbox_SessionContext.h"
#include "jni_native_object.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_SessionContext
 * Method:    SessionContext_SetSessionId
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_SessionContext_SessionContext_1SetSessionId(
    JNIEnv *env, jobject j_this, jstring j_session_id) {
  auto n_session_context =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::SessionContext>(
          env, j_this);
  if (n_session_context == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_session_context->SetSessionId(modelbox::jstring2string(env, j_session_id));
}

/*
 * Class:     com_modelbox_SessionContext
 * Method:    SessionContext_GetSessionId
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_SessionContext_SessionContext_1GetSessionId(JNIEnv *env,
                                                              jobject j_this) {
  auto n_session_context =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::SessionContext>(
          env, j_this);
  if (n_session_context == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_session_context->GetSessionId().c_str());
}

/*
 * Class:     com_modelbox_SessionContext
 * Method:    SessionContext_GetConfiguration
 * Signature: ()Lcom/modelbox/Configuration;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_SessionContext_SessionContext_1GetConfiguration(
    JNIEnv *env, jobject j_this) {
  auto n_session_context =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::SessionContext>(
          env, j_this);
  if (n_session_context == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_config = n_session_context->GetConfig();
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "configuration is invalid");
    return nullptr;
  }

  auto *j_config = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/Configuration", n_config);
  if (j_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError,
                               "configuration is invalid");
  }
  return j_config;
}

/*
 * Class:     com_modelbox_SessionContext
 * Method:    SessionContext_SetError
 * Signature: (Lcom/modelbox/FlowUnitError;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_SessionContext_SessionContext_1SetError(JNIEnv *env,
                                                          jobject j_this,
                                                          jobject j_error) {
  auto n_session_context =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::SessionContext>(
          env, j_this);
  if (n_session_context == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_error =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitError>(
          env, j_error);
  if (n_error == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "flowunit error is invalid");
    return;
  }

  n_session_context->SetError(n_error);
}

/*
 * Class:     com_modelbox_SessionContext
 * Method:    SessionContext_GetError
 * Signature: ()Lcom/modelbox/FlowUnitError;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_SessionContext_SessionContext_1GetError(JNIEnv *env,
                                                          jobject j_this) {
  auto n_session_context =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::SessionContext>(
          env, j_this);
  if (n_session_context == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_error = n_session_context->GetError();
  if (n_error == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "configuration is invalid");
    return nullptr;
  }

  auto *j_error = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/FlowUnitError", n_error);
  if (j_error == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError,
                               "configuration is invalid");
  }

  return j_error;
}
