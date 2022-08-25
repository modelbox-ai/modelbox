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

#include "com_modelbox_FlowStreamIO.h"
#include "jni_native_object.h"
#include "modelbox/flow_stream_io.h"
#include "scoped_jvm.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_FlowStreamIO
 * Method:    FlowStreamIO_CreateBuffer
 * Signature: ()Lcom/modelbox/Buffer;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_FlowStreamIO_FlowStreamIO_1CreateBuffer(JNIEnv *env,
                                                          jobject j_this) {
  auto n_stream_io =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowStreamIO>(
          env, j_this);
  if (n_stream_io == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_buff = n_stream_io->CreateBuffer();
  if (n_buff == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_NOMEM,
                               "create buffer list failed.");
    return nullptr;
  }

  auto *j_buffer =
      modelbox::JNINativeObject::NewJObject(env, "com/modelbox/Buffer", n_buff);
  if (j_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_buffer;
}

/*
 * Class:     com_modelbox_FlowStreamIO
 * Method:    FlowStreamIO_Send
 * Signature: (Ljava/lang/String;Lcom/modelbox/Buffer;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowStreamIO_FlowStreamIO_1Send__Ljava_lang_String_2Lcom_modelbox_Buffer_2(
    JNIEnv *env, jobject j_this, jstring j_inport_name, jobject j_buffer) {
  if (j_inport_name == nullptr || j_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_stream_io =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowStreamIO>(
          env, j_this);
  if (n_stream_io == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_buffer);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret =
      n_stream_io->Send(modelbox::jstring2string(env, j_inport_name), n_buffer);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_FlowStreamIO
 * Method:    FlowStreamIO_Send
 * Signature: (Ljava/lang/String;[B)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowStreamIO_FlowStreamIO_1Send__Ljava_lang_String_2_3B(
    JNIEnv *env, jobject j_this, jstring j_inport_name,
    jbyteArray j_data_array) {
  if (j_inport_name == nullptr || j_data_array == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_stream_io =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowStreamIO>(
          env, j_this);
  if (n_stream_io == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto j_data_len = env->GetArrayLength(j_data_array);
  if (j_data_len <= 0) {
    return;
  }

  jbyte *j_data_ptr = env->GetByteArrayElements(j_data_array, nullptr);
  if (j_data_ptr == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "Buffer data array is invalid");
    return;
  }
  Defer { env->ReleaseByteArrayElements(j_data_array, j_data_ptr, (jint)0); };

  auto n_buffer = n_stream_io->CreateBuffer();
  auto ret = n_buffer->BuildFromHost(j_data_ptr, j_data_len);
  if (ret != modelbox::STATUS_SUCCESS) {
    modelbox::ModelBoxJNIThrow(env, ret);
    return;
  }

  ret =
      n_stream_io->Send(modelbox::jstring2string(env, j_inport_name), n_buffer);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_FlowStreamIO
 * Method:    FlowStreamIO_Recv
 * Signature: (Ljava/lang/String;J)Lcom/modelbox/Buffer;
 */
JNIEXPORT jobject JNICALL Java_com_modelbox_FlowStreamIO_FlowStreamIO_1Recv(
    JNIEnv *env, jobject j_this, jstring j_outport_name, jlong j_timeout) {
  auto n_stream_io =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowStreamIO>(
          env, j_this);
  if (n_stream_io == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  std::shared_ptr<modelbox::Buffer> n_buff;
  auto ret = n_stream_io->Recv(modelbox::jstring2string(env, j_outport_name),
                               n_buff, (int64_t)j_timeout);
  if (ret != modelbox::STATUS_SUCCESS) {
    if (ret == modelbox::STATUS_EOF) {
      return nullptr;
    }

    modelbox::ModelBoxJNIThrow(env, ret, "recv buffer failed.");
    return nullptr;
  }

  auto *j_buffer =
      modelbox::JNINativeObject::NewJObject(env, "com/modelbox/Buffer", n_buff);
  if (j_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_buffer;
}

/*
 * Class:     com_modelbox_FlowStreamIO
 * Method:    FlowStreamIO_CloseInput
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_modelbox_FlowStreamIO_FlowStreamIO_1CloseInput(
    JNIEnv *env, jobject j_this) {
  auto n_stream_io =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowStreamIO>(
          env, j_this);
  if (n_stream_io == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_stream_io->CloseInput();
}