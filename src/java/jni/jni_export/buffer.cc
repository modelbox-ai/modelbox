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

#include "modelbox/buffer.h"

#include <memory>

#include "com_modelbox_Buffer.h"
#include "jni_native_object.h"
#include "modelbox/buffer.h"
#include "securec.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferBuild
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferBuild__J(JNIEnv *env,
                                                               jobject j_this,
                                                               jlong j_size) {
  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret = n_buffer->Build(j_size);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferBuild
 * Signature: ([B)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferBuild___3B(
    JNIEnv *env, jobject j_this, jbyteArray j_data_array) {
  if (j_data_array == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
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

  auto ret = n_buffer->BuildFromHost(j_data_ptr, j_data_len);
  if (ret != modelbox::STATUS_SUCCESS) {
    modelbox::ModelBoxJNIThrow(env, ret);
    return;
  }

  ret = n_buffer->Build(j_data_ptr, j_data_len);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetData
 * Signature: ()[B
 */
JNIEXPORT jbyteArray JNICALL
Java_com_modelbox_Buffer_BufferGetData(JNIEnv *env, jobject j_this) {
  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  const void *n_buffer_ptr = n_buffer->ConstData();
  if (n_buffer_ptr == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_NOBUFS,
                               "Buffer data is null");
    return nullptr;
  }

  int n_buffer_len = n_buffer->GetBytes();
  auto *j_data_array = env->NewByteArray(n_buffer_len);
  if (j_data_array == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_NOMEM,
                               "alloc memory for Buffer byte failed.");
    return nullptr;
  }

  env->SetByteArrayRegion(j_data_array, 0, n_buffer_len, (jbyte *)n_buffer_ptr);
  return j_data_array;
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferHasError
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL
Java_com_modelbox_Buffer_BufferHasError(JNIEnv *env, jobject j_this) {
  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return false;
  }

  return (jboolean)n_buffer->HasError();
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferSetError
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferSetError(
    JNIEnv *env, jobject j_this, jstring j_code, jstring j_message) {
  if (j_code == nullptr || j_message == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_buffer->SetError(modelbox::jstring2string(env, j_code),
                     modelbox::jstring2string(env, j_message));
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetErrorCode
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_Buffer_BufferGetErrorCode(JNIEnv *env, jobject j_this) {
  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_buffer->GetErrorCode().c_str());
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetErrorMsg
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_Buffer_BufferGetErrorMsg(JNIEnv *env, jobject j_this) {
  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_buffer->GetErrorMsg().c_str());
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetBytes
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_modelbox_Buffer_BufferGetBytes(JNIEnv *env, jobject j_this) {
  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  return (jlong)n_buffer->GetBytes();
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferCopyMeta
 * Signature: (Lcom/modelbox/Buffer;Z)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferCopyMeta(
    JNIEnv *env, jobject j_this, jobject j_buffer, jboolean j_is_overwrite) {
  if (j_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_buffer_other =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_buffer);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret = n_buffer->CopyMeta(n_buffer_other, j_is_overwrite);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetDevice
 * Signature: ()Lcom/modelbox/Device;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_Buffer_BufferGetDevice(JNIEnv *env, jobject j_this) {
  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_device = n_buffer->GetDevice();
  auto *j_device = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/Device", n_device);
  if (j_device == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_device;
}
