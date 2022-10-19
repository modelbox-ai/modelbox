
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

#include <modelbox/base/log.h>
#include <modelbox/flowunit.h>

#include <memory>

#include "com_modelbox_FlowUnitDesc.h"
#include "jni_native_object.h"
#include "scoped_jvm.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescNew(JNIEnv *env, jobject j_this) {
  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::FlowUnitDesc>());
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescGetFlowUnitName
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescGetFlowUnitName(JNIEnv *env,
                                                           jobject j_this) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_desc->GetFlowUnitName().c_str());
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescGetFlowUnitType
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescGetFlowUnitType(JNIEnv *env,
                                                           jobject j_this) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return env->NewStringUTF(n_desc->GetFlowUnitType().c_str());
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescGetFlowUnitAliasName
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescGetFlowUnitAliasName(
    JNIEnv *env, jobject j_this) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return nullptr;
  }

  return env->NewStringUTF(n_desc->GetFlowUnitAliasName().c_str());
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescGetFlowUnitArgument
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescGetFlowUnitArgument(JNIEnv *env,
                                                               jobject j_this) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return nullptr;
  }

  return env->NewStringUTF(n_desc->GetFlowUnitArgument().c_str());
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetFlowUnitName
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetFlowUnitName(JNIEnv *env,
                                                           jobject j_this,
                                                           jstring j_name) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  n_desc->SetFlowUnitName(modelbox::jstring2string(env, j_name));
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetFlowUnitType
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetFlowUnitType(JNIEnv *env,
                                                           jobject j_this,
                                                           jstring j_type) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  n_desc->SetFlowUnitType(modelbox::jstring2string(env, j_type));
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescAddFlowUnitInput
 * Signature: (Lcom/modelbox/FlowUnitInput;)V;
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescAddFlowUnitInput(JNIEnv *env,
                                                            jobject j_this,
                                                            jobject j_input) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  auto n_input =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitInput>(
          env, j_input);
  if (n_input == nullptr) {
    return;
  }

  auto status = n_desc->AddFlowUnitInput(*n_input);
  modelbox::ModelBoxJNIThrow(env, status);
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescAddFlowUnitOutput
 * Signature: (Lcom/modelbox/FlowUnitOutput;)V;
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescAddFlowUnitOutput(JNIEnv *env,
                                                             jobject j_this,
                                                             jobject j_output) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  auto n_output =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitOutput>(
          env, j_output);
  if (n_output == nullptr) {
    return;
  }

  auto status = n_desc->AddFlowUnitOutput(*n_output);
  modelbox::ModelBoxJNIThrow(env, status);
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescAddFlowUnitOption
 * Signature: (Lcom/modelbox/FlowUnitOption;)Lcom/modelbox/Status;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescAddFlowUnitOption(JNIEnv *env,
                                                             jobject j_this,
                                                             jobject j_option) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return nullptr;
  }

  auto n_option =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitOption>(
          env, j_option);
  if (n_option == nullptr) {
    return nullptr;
  }

  auto status = n_desc->AddFlowUnitOption(*n_option);
  return nullptr;
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetConditionType
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetConditionType(JNIEnv *env,
                                                            jobject j_this,
                                                            jlong j_type) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  if (j_type > modelbox::IF_ELSE) {
    return;
  }

  n_desc->SetConditionType(static_cast<modelbox::ConditionType>(j_type));
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetLoopType
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetLoopType(
    JNIEnv *env, jobject j_this, jlong j_type) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  if (j_type > modelbox::LOOP) {
    return;
  }

  n_desc->SetLoopType(static_cast<modelbox::LoopType>(j_type));
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetOutputType
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetOutputType(
    JNIEnv *env, jobject j_this, jlong j_type) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  if (j_type > modelbox::COLLAPSE) {
    return;
  }

  n_desc->SetOutputType(static_cast<modelbox::FlowOutputType>(j_type));
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetFlowType
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetFlowType(
    JNIEnv *env, jobject j_this, jlong j_type) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  if (j_type > modelbox::NORMAL) {
    return;
  }

  n_desc->SetFlowType(static_cast<modelbox::FlowType>(j_type));
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetStreamSameCount
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetStreamSameCount(
    JNIEnv *env, jobject j_this, jboolean j_same_count) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  n_desc->SetStreamSameCount(j_same_count);
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetInputContiguous
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetInputContiguous(
    JNIEnv *env, jobject j_this, jboolean j_contiguous) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  n_desc->SetInputContiguous(j_contiguous);
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetResourceNice
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetResourceNice(JNIEnv *env,
                                                           jobject j_this,
                                                           jboolean j_nice) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  n_desc->SetResourceNice(j_nice);
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetCollapseAll
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetCollapseAll(
    JNIEnv *env, jobject j_this, jboolean j_collapse_all) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  n_desc->SetCollapseAll(j_collapse_all);
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetExceptionVisible
 * Signature: (Z)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetExceptionVisible(
    JNIEnv *env, jobject j_this, jboolean j_visible) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  n_desc->SetExceptionVisible(j_visible);
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetDescription
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetDescription(JNIEnv *env,
                                                          jobject j_this,
                                                          jstring j_desc) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  auto s_desc = modelbox::jstring2string(env, j_desc);
  n_desc->SetDescription(s_desc);
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetMaxBatchSize
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetMaxBatchSize(
    JNIEnv *env, jobject j_this, jlong j_max_batch_size) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  n_desc->SetMaxBatchSize(j_max_batch_size);
}

/*
 * Class:     com_modelbox_FlowUnitDesc
 * Method:    FlowUnitDescSetDefaultBatchSize
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_FlowUnitDesc_FlowUnitDescSetDefaultBatchSize(
    JNIEnv *env, jobject j_this, jlong j_default_batch_size) {
  auto n_desc =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitDesc>(
          env, j_this);
  if (n_desc == nullptr) {
    return;
  }

  n_desc->SetDefaultBatchSize(j_default_batch_size);
}
