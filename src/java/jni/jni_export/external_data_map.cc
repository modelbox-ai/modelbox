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

#include "modelbox/external_data_map.h"

#include <memory>

#include "com_modelbox_ExternalDataMap.h"
#include "jni_native_object.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_ExternalDataMap
 * Method:    ExternalDataMap_CreateBufferList
 * Signature: ()Lcom/modelbox/BufferList;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_ExternalDataMap_ExternalDataMap_1CreateBufferList(
    JNIEnv *env, jobject j_this) {
  auto n_datamap =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_this);
  if (n_datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_bufflist = n_datamap->CreateBufferList();
  if (n_bufflist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_NOMEM,
                               "create buffer list failed.");
    return nullptr;
  }

  auto *j_buffer_list = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/BufferList", n_bufflist);
  if (j_buffer_list == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_buffer_list;
}

/*
 * Class:     com_modelbox_ExternalDataMap
 * Method:    ExternalDataMap_SetOutputMeta
 * Signature: (Ljava/lang/String;Lcom/modelbox/DataMeta;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_ExternalDataMap_ExternalDataMap_1SetOutputMeta(
    JNIEnv *env, jobject j_this, jstring j_name, jobject j_data_meta) {
  if (j_name == nullptr || j_data_meta) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_datamap =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_this);
  if (n_datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_data_meta =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataMeta>(
          env, j_data_meta);
  if (n_data_meta == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret = n_datamap->SetOutputMeta(modelbox::jstring2string(env, j_name),
                                      n_data_meta);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_ExternalDataMap
 * Method:    ExternalDataMap_Send
 * Signature: (Ljava/lang/String;Lcom/modelbox/BufferList;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ExternalDataMap_ExternalDataMap_1Send(
    JNIEnv *env, jobject j_this, jstring j_port_name, jobject j_bufferlist) {
  if (j_port_name == nullptr || j_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_datamap =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_this);
  if (n_datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto j_buffer_list =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_bufferlist);
  if (j_buffer_list == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret = n_datamap->Send(modelbox::jstring2string(env, j_port_name),
                             j_buffer_list);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_ExternalDataMap
 * Method:    ExternalDataMap_Recv
 * Signature: (J)Ljava/util/HashMap;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_ExternalDataMap_ExternalDataMap_1Recv(JNIEnv *env,
                                                        jobject j_this,
                                                        jlong j_timeout) {
  auto n_datamap =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_this);
  if (n_datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  modelbox::OutputBufferList map_buffer_list;
  auto ret = n_datamap->Recv(map_buffer_list, (int32_t)j_timeout);
  if (ret != modelbox::STATUS_SUCCESS) {
    if (ret == modelbox::STATUS_EOF) {
      return nullptr;
    }

    modelbox::ModelBoxJNIThrow(env, ret);
    return nullptr;
  }

  jclass j_map_cls = env->FindClass("java/util/HashMap");
  if (j_map_cls == nullptr) {
    ret = {modelbox::STATUS_INTERNAL, "cannot found hash map class"};
    modelbox::ModelBoxJNIThrow(env, ret);
    return nullptr;
  }

  Defer { env->DeleteLocalRef(j_map_cls); };

  jmethodID init = env->GetMethodID(j_map_cls, "<init>", "()V");
  jobject j_hashmap = env->NewObject(j_map_cls, init, 10);
  jmethodID put = env->GetMethodID(
      j_map_cls, "put",
      "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

  for (const auto &item : map_buffer_list) {
    jstring j_key = env->NewStringUTF(item.first.c_str());
    auto *j_bufflist = modelbox::JNINativeObject::NewJObject(
        env, "com/modelbox/BufferList", item.second);
    env->CallObjectMethod(j_hashmap, put, j_key, j_bufflist);
    env->DeleteLocalRef(j_key);
    env->DeleteLocalRef(j_bufflist);
  }

  return j_hashmap;
}

/*
 * Class:     com_modelbox_ExternalDataMap
 * Method:    ExternalDataMap_Close
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ExternalDataMap_ExternalDataMap_1Close(
    JNIEnv *env, jobject j_this) {
  auto n_datamap =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_this);
  if (n_datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_datamap->Close();
}

/*
 * Class:     com_modelbox_ExternalDataMap
 * Method:    ExternalDataMap_Shutdown
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_ExternalDataMap_ExternalDataMap_1Shutdown(JNIEnv *env,
                                                            jobject j_this) {
  auto n_datamap =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_this);
  if (n_datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_datamap->Shutdown();
}

/*
 * Class:     com_modelbox_ExternalDataMap
 * Method:    ExternalDataMap_GetSessionContext
 * Signature: ()Lcom/modelbox/SessionContext;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_ExternalDataMap_ExternalDataMap_1GetSessionContext(
    JNIEnv *env, jobject j_this) {
  auto n_datamap =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_this);
  if (n_datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_session_context = n_datamap->GetSessionContext();
  if (n_session_context == nullptr) {
    return nullptr;
  }

  auto *j_session_ctx = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/SessionContext", n_session_context);
  if (j_session_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_session_ctx;
}

/*
 * Class:     com_modelbox_ExternalDataMap
 * Method:    ExternalDataMap_GetSessionConfig
 * Signature: ()Lcom/modelbox/Configuration;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_ExternalDataMap_ExternalDataMap_1GetSessionConfig(
    JNIEnv *env, jobject j_this) {
  auto n_datamap =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_this);
  if (n_datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_session_config = n_datamap->GetSessionConfig();
  if (n_session_config == nullptr) {
    return nullptr;
  }

  auto *j_session_config = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/SessionConfig", n_session_config);
  if (j_session_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_session_config;
}

/*
 * Class:     com_modelbox_ExternalDataMap
 * Method:    ExternalDataMap_GetLastError
 * Signature: ()Lcom/modelbox/FlowUnitError;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_ExternalDataMap_ExternalDataMap_1GetLastError(
    JNIEnv *env, jobject j_this) {
  auto n_datamap =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_this);
  if (n_datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_laste_error = n_datamap->GetLastError();
  if (n_laste_error == nullptr) {
    return nullptr;
  }

  auto *j_last_error = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/FlowUnitError", n_laste_error);
  if (j_last_error == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_last_error;
}
