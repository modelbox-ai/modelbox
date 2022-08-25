
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

#include "com_modelbox_DataContext.h"
#include "jni_native_object.h"
#include "modelbox/data_context.h"
#include "scoped_jvm.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_Input
 * Signature: (Ljava/lang/String;)Lcom/modelbox/BufferList;
 */
JNIEXPORT jobject JNICALL Java_com_modelbox_DataContext_DataContext_1Input(
    JNIEnv *env, jobject j_this, jstring j_portname) {
  if (j_portname == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return nullptr;
  }

  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_buffer_list =
      n_data_ctx->Input(modelbox::jstring2string(env, j_portname));
  if (n_buffer_list) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input port not exists");
    return nullptr;
  }

  auto *j_buffer_list = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/BufferList", n_buffer_list);
  if (j_buffer_list == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_buffer_list;
}

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_Output
 * Signature: (Ljava/lang/String;)Lcom/modelbox/BufferList;
 */
JNIEXPORT jobject JNICALL Java_com_modelbox_DataContext_DataContext_1Output(
    JNIEnv *env, jobject j_this, jstring j_portname) {
  if (j_portname == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return nullptr;
  }

  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_buffer_list =
      n_data_ctx->Output(modelbox::jstring2string(env, j_portname));
  if (n_buffer_list) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "output port not exists");
    return nullptr;
  }

  auto *j_buffer_list = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/BufferList", n_buffer_list);
  if (j_buffer_list == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_buffer_list;
}

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_External
 * Signature: ()Lcom/modelbox/BufferList;
 */
JNIEXPORT jobject JNICALL Java_com_modelbox_DataContext_DataContext_1External(
    JNIEnv *env, jobject j_this) {
  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_buffer_list = n_data_ctx->External();
  if (n_buffer_list) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "external port not exists");
    return nullptr;
  }

  auto *j_buffer_list = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/BufferList", n_buffer_list);
  if (j_buffer_list == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_buffer_list;
}

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_HasError
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_com_modelbox_DataContext_DataContext_1HasError(
    JNIEnv *env, jobject j_this) {
  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return false;
  }

  return n_data_ctx->HasError();
}

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_SendEvent
 * Signature: (Lcom/modelbo::DataContext;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_DataContext_DataContext_1SendEvent(
    JNIEnv *env, jobject j_this, jobject j_event) {
  if (j_event == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_event =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitEvent>(
          env, j_event);
  if (n_event == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_data_ctx->SendEvent(n_event);
}

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_SetPrivate
 * Signature: (Ljava/lang/String;Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_DataContext_DataContext_1SetPrivate(
    JNIEnv *env, jobject j_this, jstring j_key, jobject j_object) {
  if (j_object == nullptr || j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto *j_global_object = env->NewGlobalRef(j_object);
  std::shared_ptr<void> priv_ptr(
      (void *)j_global_object, [](void *j_global_object) {
        modelbox::ScopedJvm scoped;
        scoped.GetJNIEnv()->DeleteGlobalRef((jobject)j_global_object);
      });
  n_data_ctx->SetPrivate(modelbox::jstring2string(env, j_key), priv_ptr);
}

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_GetPrivate
 * Signature: (Ljava/lang/String;)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_com_modelbox_DataContext_DataContext_1GetPrivate(
    JNIEnv *env, jobject j_this, jstring j_key) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return nullptr;
  }

  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_object = n_data_ctx->GetPrivate(modelbox::jstring2string(env, j_key));
  if (n_object == nullptr) {
    return nullptr;
  }

  return (jobject)n_object.get();
}

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_GetInputMeta
 * Signature: (Ljava/lang/String;)Lcom/modelbox/DataMeta;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_DataContext_DataContext_1GetInputMeta(JNIEnv *env,
                                                        jobject j_this,
                                                        jstring j_portname) {
  if (j_portname == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return nullptr;
  }

  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_datameta =
      n_data_ctx->GetInputMeta(modelbox::jstring2string(env, j_portname));
  if (n_datameta == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "port meta not exists");
    return nullptr;
  }

  auto *j_datameta = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/DataMeta", n_datameta);
  if (j_datameta == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
  }

  return j_datameta;
}

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_SetOututMeta
 * Signature: (Ljava/lang/String;Lcom/modelbox/DataMeta;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_DataContext_DataContext_1SetOututMeta(
    JNIEnv *env, jobject j_this, jstring j_portname, jobject j_datameta) {
  if (j_portname == nullptr || j_datameta == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_datameta =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataMeta>(
          env, j_datameta);
  if (n_datameta == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_data_ctx->SetOutputMeta(modelbox::jstring2string(env, j_portname),
                            n_datameta);
}

/*
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_GetSessionContext
 * Signature: ()Lcom/modelbox/SessionContext;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_DataContext_DataContext_1GetSessionContext(JNIEnv *env,
                                                             jobject j_this) {
  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_session_config = n_data_ctx->GetSessionConfig();
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
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_GetSessionConfig
 * Signature: ()Lcom/modelbox/Configuration;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_DataContext_DataContext_1GetSessionConfig(JNIEnv *env,
                                                            jobject j_this) {
  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_session_context = n_data_ctx->GetSessionContext();
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
 * Class:     com_modelbox_DataContext
 * Method:    DataContext_GetStatistics
 * Signature: ()Lcom/modelbox/StatisticsItem;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_DataContext_DataContext_1GetStatistics(JNIEnv *env,
                                                         jobject j_this) {
  auto n_data_ctx =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataContext>(
          env, j_this);
  if (n_data_ctx == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_statistics = n_data_ctx->GetStatistics();
  if (n_statistics == nullptr) {
    return nullptr;
  }

  auto *j_statistics = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/StatisticsItem", n_statistics);
  if (j_statistics == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_statistics;
}