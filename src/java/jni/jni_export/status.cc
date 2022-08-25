
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

#include "modelbox/base/status.h"

#include <memory>

#include "com_modelbox_Status.h"
#include "jni_native_object.h"
#include "modelbox/base/utils.h"
#include "throw.h"
#include "utils.h"

jobject GetJStatusCodeFromStatus(JNIEnv *env, modelbox::Status &status) {
  jclass j_cls = env->FindClass("com/modelbox/StatusCode");
  if (j_cls == nullptr) {
    modelbox::StatusError = {modelbox::STATUS_INTERNAL,
                             "Cannot find class StatusCode"};
    return nullptr;
  }

  Defer { env->DeleteLocalRef(j_cls); };

  jfieldID j_field = env->GetStaticFieldID(
      j_cls, status.StrStatusCode().c_str(), "Lcom/modelbox/StatusCode;");
  if (j_field == nullptr) {
    modelbox::StatusError = {modelbox::STATUS_FAULT,
                             "Cannot find enum for StatusCode"};
    return nullptr;
  }

  jobject j_code = env->GetStaticObjectField(j_cls, j_field);

  return j_code;
}

/*
 * Class:     com_modelbox_Status
 * Method:    StatusNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_Status_StatusNew(JNIEnv *env,
                                                           jobject j_this) {
  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::Status>());
}

/*
 * Class:     com_modelbox_Status
 * Method:    StatusSetCode
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Status_StatusSetCode(JNIEnv *env,
                                                              jobject j_this,
                                                              jlong j_code) {
  auto n_status =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(env,
                                                                      j_this);
  if (n_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  if (j_code >= modelbox::STATUS_LASTFLAG) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "statuscode is invalid");
    return;
  }

  *n_status = (modelbox::StatusCode)j_code;
}

/*
 * Class:     com_modelbox_Status
 * Method:    StatusWrap
 * Signature: (Lcom/modelbox/Status;JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Status_StatusWrap(
    JNIEnv *env, jobject j_this, jobject j_status_other, jlong j_code,
    jstring j_message) {
  auto n_status =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(env,
                                                                      j_this);
  if (n_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_status_other =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(
          env, j_status_other);
  if (n_status_other == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  if (j_code >= modelbox::STATUS_LASTFLAG) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "statuscode is invalid");
    return;
  }

  n_status->Wrap(*n_status_other, (modelbox::StatusCode)j_code,
                 modelbox::jstring2string(env, j_message));
}

/*
 * Class:     com_modelbox_Status
 * Method:    StatusToSting
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_Status_StatusToSting(JNIEnv *env, jobject j_this) {
  auto n_status =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(env,
                                                                      j_this);
  if (n_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_status->ToString().c_str());
}

/*
 * Class:     com_modelbox_Status
 * Method:    StatusCode
 * Signature: ()Lcom/modelbox/StatusCode;
 */
JNIEXPORT jobject JNICALL Java_com_modelbox_Status_StatusCode(JNIEnv *env,
                                                              jobject j_this) {
  auto n_status =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(env,
                                                                      j_this);
  if (n_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }
  auto *j_code = GetJStatusCodeFromStatus(env, *n_status);
  if (j_code == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
  }

  return j_code;
}

/*
 * Class:     com_modelbox_Status
 * Method:    StatusStrCode
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_Status_StatusStrCode(JNIEnv *env, jobject j_this) {
  auto n_status =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(env,
                                                                      j_this);
  if (n_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_status->StrCode().c_str());
}

/*
 * Class:     com_modelbox_Status
 * Method:    StatusSetErrorMsg
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Status_StatusSetErrorMsg(
    JNIEnv *env, jobject j_this, jstring j_message) {
  auto n_status =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(env,
                                                                      j_this);
  if (n_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_status->SetErrormsg(modelbox::jstring2string(env, j_message));
}

/*
 * Class:     com_modelbox_Status
 * Method:    StatusErrorMsg
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_Status_StatusErrorMsg(JNIEnv *env, jobject j_this) {
  auto n_status =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(env,
                                                                      j_this);
  if (n_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_status->Errormsg().c_str());
}

/*
 * Class:     com_modelbox_Status
 * Method:    StatusWrapErrormsgs
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_Status_StatusWrapErrormsgs(JNIEnv *env, jobject j_this) {
  auto n_status =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(env,
                                                                      j_this);
  if (n_status == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_status->WrapErrormsgs().c_str());
}