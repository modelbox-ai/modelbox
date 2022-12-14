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

#include "com_modelbox_BufferList.h"
#include "jni_native_object.h"
#include "modelbox/buffer_list.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListBuild
 * Signature: ([I)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_BufferList_BufferListBuild(
    JNIEnv *env, jobject j_this, jintArray j_size_list) {
  if (j_size_list == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  jsize j_size_list_len = env->GetArrayLength(j_size_list);
  if (j_size_list_len <= 0) {
    return;
  }

  jint *j_size_list_data = env->GetIntArrayElements(j_size_list, nullptr);
  if (j_size_list_data == nullptr) {
    modelbox::Status ret = {modelbox::STATUS_NOMEM,
                            "Get Buffer list array element failed."};
    modelbox::ModelBoxJNIThrow(env, ret);
    return;
  }

  std::vector<size_t> size_list;
  size_list.reserve(j_size_list_len);
  for (int i = 0; i < j_size_list_len; i++) {
    size_list.push_back(j_size_list_data[i]);
  }

  env->ReleaseIntArrayElements(j_size_list, j_size_list_data, (jint)0);

  auto ret = n_bufferlist->Build(size_list, true);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListAt
 * Signature: (J)Lcom/modelbox/Buffer;
 */
JNIEXPORT jobject JNICALL Java_com_modelbox_BufferList_BufferListAt(
    JNIEnv *env, jobject j_this, jlong j_index) {
  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_buffer = n_bufferlist->At((size_t)j_index);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_RANGE,
                               "Bufferlist index is invalid.");
    return nullptr;
  }

  auto *j_buffer = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/Buffer", n_buffer);
  if (j_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_buffer;
}

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListSize
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_modelbox_BufferList_BufferListSize(JNIEnv *env, jobject j_this) {
  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  return (jlong)n_bufferlist->Size();
}

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListPushBack
 * Signature: (Lcom/modelbox/Buffer;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_BufferList_BufferListPushBack__Lcom_modelbox_Buffer_2(
    JNIEnv *env, jobject j_this, jobject j_buffer) {
  if (j_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
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

  n_bufferlist->PushBack(n_buffer);
}

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListPushBack
 * Signature: ([B)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_BufferList_BufferListPushBack___3B(
    JNIEnv *env, jobject j_this, jbyteArray j_data_array) {
  if (j_data_array == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
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

  auto n_buffer = std::make_shared<modelbox::Buffer>(n_bufferlist->GetDevice());
  auto ret = n_buffer->BuildFromHost(j_data_ptr, j_data_len);
  if (ret != modelbox::STATUS_SUCCESS) {
    modelbox::ModelBoxJNIThrow(env, ret);
    return;
  }

  n_bufferlist->PushBack(n_buffer);
}

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListAssign
 * Signature: ([Lcom/modelbox/Buffer;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_BufferList_BufferListAssign(
    JNIEnv *env, jobject j_this, jobjectArray j_buffer_list) {
  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_NOTSUPPORT, "not supported");
}

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListGetData
 * Signature: ()[B
 */
JNIEXPORT jbyteArray JNICALL
Java_com_modelbox_BufferList_BufferListGetData(JNIEnv *env, jobject j_this) {
  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto *j_data = env->NewByteArray(n_bufferlist->GetBytes());
  if (j_data == nullptr) {
    modelbox::Status ret = {modelbox::STATUS_NOMEM,
                            "alloc memory for buffer list data failed."};
    modelbox::ModelBoxJNIThrow(env, ret);
    return nullptr;
  }

  env->SetByteArrayRegion(j_data, 0, n_bufferlist->GetBytes(),
                          (jbyte *)n_bufferlist->ConstData());
  return j_data;
}

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListGetDirectData
 * Signature: ()Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_BufferList_BufferListGetDirectData__(JNIEnv *env,
                                                       jobject j_this) {
  bool is_const = false;
  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  void *n_buffer_ptr = n_bufferlist->MutableData();
  if (n_buffer_ptr == nullptr) {
    n_buffer_ptr = (void *)n_bufferlist->ConstData();
    if (n_buffer_ptr == nullptr) {
      modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                                 "buffer list data is null");
      return nullptr;
    }
    is_const = true;
  }

  auto *j_byte_buffer =
      env->NewDirectByteBuffer(n_buffer_ptr, n_bufferlist->GetBytes());
  if (j_byte_buffer == nullptr) {
    modelbox::Status ret = {modelbox::STATUS_NOMEM,
                            "alloc memory for buffer list data failed."};
    modelbox::ModelBoxJNIThrow(env, ret);
    return nullptr;
  }

  if (is_const == false) {
    return j_byte_buffer;
  }

  jmethodID asreadonly_method =
      env->GetMethodID(env->GetObjectClass(j_byte_buffer), "asReadOnlyBuffer",
                       "()Ljava/nio/ByteBuffer;");
  if (asreadonly_method == nullptr) {
    modelbox::Status ret = {modelbox::STATUS_NOMEM,
                            "get asreadonly method failed."};
    modelbox::ModelBoxJNIThrow(env, ret);
    return nullptr;
  }

  jobject j_readonly_byte_buffer =
      env->CallObjectMethod(j_byte_buffer, asreadonly_method);
  return j_readonly_byte_buffer;
}

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListGetDirectData
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_BufferList_BufferListGetDirectData__J(JNIEnv *env,
                                                        jobject j_this,
                                                        jlong j_index) {
  bool is_const = false;
  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  void *n_buffer_ptr = n_bufferlist->MutableBufferData(j_index);
  if (n_buffer_ptr == nullptr) {
    n_buffer_ptr = (void *)n_bufferlist->ConstBufferData(j_index);
    if (n_buffer_ptr == nullptr) {
      if (n_bufferlist->GetBytes() != 0) {
        modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                                   "buffer is not continuous.");
        return nullptr;
      }

      modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                                 "buffer list data is null");
      return nullptr;
    }
    is_const = true;
  }

  //TODO The reference to n_bufferlist should be added to avoid dangling pointers of j_byte_buffer
  //https://stackoverflow.com/questions/46844275/freeing-memory-wrapped-with-newdirectbytebuffer
  auto *j_byte_buffer =
      env->NewDirectByteBuffer(n_buffer_ptr, n_bufferlist->GetBytes());
  if (j_byte_buffer == nullptr) {
    modelbox::Status ret = {modelbox::STATUS_NOMEM,
                            "alloc memory for buffer list data failed."};
    modelbox::ModelBoxJNIThrow(env, ret);
    return nullptr;
  }

  if (is_const == false) {
    return j_byte_buffer;
  }

  jmethodID asreadonly_method =
      env->GetMethodID(env->GetObjectClass(j_byte_buffer), "asReadOnlyBuffer",
                       "()Ljava/nio/ByteBuffer;");
  if (asreadonly_method == nullptr) {
    modelbox::Status ret = {modelbox::STATUS_NOMEM,
                            "get asreadonly method failed."};
    modelbox::ModelBoxJNIThrow(env, ret);
    return nullptr;
  }

  jobject j_readonly_byte_buffer =
      env->CallObjectMethod(j_byte_buffer, asreadonly_method);
  return j_readonly_byte_buffer;
}
/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListGetDevice
 * Signature: ()Lcom/modelbox/Device;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_BufferList_BufferListGetDevice(JNIEnv *env, jobject j_this) {
  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_device = n_bufferlist->GetDevice();
  if (n_device == nullptr) {
    return nullptr;
  }

  auto *j_device = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/Device", n_device);
  if (j_device == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_device;
}

/*
 * Class:     com_modelbox_BufferList
 * Method:    BufferListReset
 * Signature: ()V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_BufferList_BufferListReset(JNIEnv *env, jobject j_this) {
  auto n_bufferlist =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::BufferList>(
          env, j_this);
  if (n_bufferlist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret = n_bufferlist->Reset();
  modelbox::ModelBoxJNIThrow(env, ret);
}
