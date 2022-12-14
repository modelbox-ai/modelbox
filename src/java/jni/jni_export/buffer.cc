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
 * Method:    BufferGetDirectData
 * Signature: ()Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_Buffer_BufferGetDirectData(JNIEnv *env, jobject j_this) {
  bool is_const = false;
  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  void *n_buffer_ptr = n_buffer->MutableData();
  if (n_buffer_ptr == nullptr) {
    n_buffer_ptr = (void *)n_buffer->ConstData();
    if (n_buffer_ptr == nullptr) {
      modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_NOBUFS,
                                 "Buffer data is null");
      return nullptr;
    }

    is_const = true;
  }

  //TODO The reference to n_bufferlist should be added to avoid dangling pointers of j_byte_buffer
  //https://stackoverflow.com/questions/46844275/freeing-memory-wrapped-with-newdirectbytebuffer
  jobject j_byte_buffer =
      env->NewDirectByteBuffer((void *)n_buffer_ptr, n_buffer->GetBytes());
  if (j_byte_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_NOMEM,
                               "alloc memory for Buffer byte failed.");
    return nullptr;
  }

  if (is_const == false) {
    return j_byte_buffer;
  }

  jmethodID asreadonly_method =
      env->GetMethodID(env->GetObjectClass(j_byte_buffer), "asReadOnlyBuffer",
                       "()Ljava/nio/ByteBuffer;");
  if (asreadonly_method == nullptr) {
    MBLOG_ERROR << "get asreadonly method failed.";
    return nullptr;
  }

  jobject j_readonly_byte_buffer =
      env->CallObjectMethod(j_byte_buffer, asreadonly_method);
  return j_readonly_byte_buffer;
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
 * Method:    BufferSetMetaLong
 * Signature: (Ljava/lang/String;J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferSetMetaLong(
    JNIEnv *env, jobject j_this, jstring j_key, jlong j_value) {
  if (j_key == nullptr) {
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

  auto n_value = (int64_t)j_value;

  n_buffer->Set(modelbox::jstring2string(env, j_key), n_value);
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferSetMetaInt
 * Signature: (Ljava/lang/String;I)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferSetMetaInt(JNIEnv *env,
                                                                 jobject j_this,
                                                                 jstring j_key,
                                                                 jint j_value) {
  if (j_key == nullptr) {
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

  auto n_value = (int32_t)j_value;

  n_buffer->Set(modelbox::jstring2string(env, j_key), n_value);
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferSetMetaString
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferSetMetaString(
    JNIEnv *env, jobject j_this, jstring j_key, jstring j_value) {
  if (j_key == nullptr || j_value == nullptr) {
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

  auto n_value = modelbox::jstring2string(env, j_value);

  n_buffer->Set(modelbox::jstring2string(env, j_key), n_value);
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferSetMetaDouble
 * Signature: (Ljava/lang/String;D)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferSetMetaDouble(
    JNIEnv *env, jobject j_this, jstring j_key, jdouble j_value) {
  if (j_key == nullptr) {
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

  auto n_value = (double)j_value;

  n_buffer->Set(modelbox::jstring2string(env, j_key), n_value);
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferSetMetaFloat
 * Signature: (Ljava/lang/String;F)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferSetMetaFloat(
    JNIEnv *env, jobject j_this, jstring j_key, jfloat j_value) {
  if (j_key == nullptr) {
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

  auto n_value = (float)j_value;

  n_buffer->Set(modelbox::jstring2string(env, j_key), n_value);
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferSetMetaBool
 * Signature: (Ljava/lang/String;Z)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Buffer_BufferSetMetaBool(
    JNIEnv *env, jobject j_this, jstring j_key, jboolean j_value) {
  if (j_key == nullptr) {
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

  auto n_value = (bool)j_value;

  n_buffer->Set(modelbox::jstring2string(env, j_key), n_value);
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetMetaLong
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_Buffer_BufferGetMetaLong(
    JNIEnv *env, jobject j_this, jstring j_key) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return 0;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  int64_t n_value = 0;
  if (n_buffer->Get(modelbox::jstring2string(env, j_key), n_value) == false) {
    modelbox::ModelBoxJNIThrow(
        env, modelbox::STATUS_INVALID,
        "key not found in meta or value type is invalid.");
    return 0;
  }

  return (jlong)n_value;
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetMetaInt
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_com_modelbox_Buffer_BufferGetMetaInt(
    JNIEnv *env, jobject j_this, jstring j_key) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return 0;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  int32_t n_value = 0;
  if (n_buffer->Get(modelbox::jstring2string(env, j_key), n_value) == false) {
    modelbox::ModelBoxJNIThrow(
        env, modelbox::STATUS_INVALID,
        "key not found in meta or value type is invalid.");
    return 0;
  }

  return (jint)n_value;
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetMetaString
 * Signature: (Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_modelbox_Buffer_BufferGetMetaString(
    JNIEnv *env, jobject j_this, jstring j_key) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return nullptr;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  std::string n_value;
  if (n_buffer->Get(modelbox::jstring2string(env, j_key), n_value) == false) {
    modelbox::ModelBoxJNIThrow(
        env, modelbox::STATUS_INVALID,
        "key not found in meta or value type is invalid.");
    return nullptr;
  }

  return env->NewStringUTF(n_value.c_str());
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetMetaDouble
 * Signature: (Ljava/lang/String;)D
 */
JNIEXPORT jdouble JNICALL Java_com_modelbox_Buffer_BufferGetMetaDouble(
    JNIEnv *env, jobject j_this, jstring j_key) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return 0;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  double n_value = 0;
  if (n_buffer->Get(modelbox::jstring2string(env, j_key), n_value) == false) {
    modelbox::ModelBoxJNIThrow(
        env, modelbox::STATUS_INVALID,
        "key not found in meta or value type is invalid.");
    return 0;
  }

  return (jdouble)n_value;
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetMetaFloat
 * Signature: (Ljava/lang/String;)F
 */
JNIEXPORT jfloat JNICALL Java_com_modelbox_Buffer_BufferGetMetaFloat(
    JNIEnv *env, jobject j_this, jstring j_key) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return 0;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  float n_value = 0;
  if (n_buffer->Get(modelbox::jstring2string(env, j_key), n_value) == false) {
    modelbox::ModelBoxJNIThrow(
        env, modelbox::STATUS_INVALID,
        "key not found in meta or value type is invalid.");
    return 0;
  }

  return (jfloat)n_value;
}

/*
 * Class:     com_modelbox_Buffer
 * Method:    BufferGetMetaBool
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_modelbox_Buffer_BufferGetMetaBool(
    JNIEnv *env, jobject j_this, jstring j_key) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return false;
  }

  auto n_buffer =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Buffer>(env,
                                                                      j_this);
  if (n_buffer == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return false;
  }

  bool n_value = false;
  if (n_buffer->Get(modelbox::jstring2string(env, j_key), n_value) == false) {
    modelbox::ModelBoxJNIThrow(
        env, modelbox::STATUS_INVALID,
        "key not found in meta or value type is invalid.");
    return false;
  }

  return (jboolean)n_value;
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
