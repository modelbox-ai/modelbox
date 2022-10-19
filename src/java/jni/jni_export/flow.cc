
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

#include "modelbox/flow.h"

#include <memory>

#include "com_modelbox_Flow.h"
#include "jni_native_object.h"
#include "modelbox/base/log.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_Flow_FlowNew(JNIEnv *env,
                                                       jobject j_this) {
  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::Flow>());
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowWait
 * Signature: (JLcom/modelbox/Status;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_modelbox_Flow_FlowWait(JNIEnv *env,
                                                           jobject j_this,
                                                           jlong j_timeout,
                                                           jobject j_status) {
  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return false;
  }

  modelbox::Status wait_ret;
  auto ret = n_flow->Wait((int64_t)j_timeout, &wait_ret);
  if (ret != modelbox::STATUS_SUCCESS) {
    if (ret == modelbox::STATUS_TIMEDOUT) {
      return false;
    }

    modelbox::ModelBoxJNIThrow(env, ret);
    return false;
  }

  if (j_status) {
    auto n_status =
        modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Status>(
            env, j_status);
    if (n_status) {
      *n_status = wait_ret;
    }
  }

  return true;
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowStartRun
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Flow_FlowStartRun(JNIEnv *env,
                                                           jobject j_this) {
  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret = n_flow->StartRun();
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowInit
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Flow_FlowInit__Ljava_lang_String_2Ljava_lang_String_2(
    JNIEnv *env, jobject j_this, jstring j_name, jstring j_graph) {
  if (j_graph == nullptr || j_name == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret = n_flow->Init(modelbox::jstring2string(env, j_name),
                          modelbox::jstring2string(env, j_graph));
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowInit
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Flow_FlowInit__Ljava_lang_String_2(
    JNIEnv *env, jobject j_this, jstring j_file) {
  if (j_file == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr || j_file == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret = n_flow->Init(modelbox::jstring2string(env, j_file));
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowInit
 * Signature:
 * (Ljava/lang/String;Lcom/modelbox/Configuration;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Flow_FlowInitByName__Ljava_lang_String_2Lcom_modelbox_Configuration_2Ljava_lang_String_2(
    JNIEnv *env, jobject j_this, jstring j_name, jobject j_args,
    jstring j_flowdir) {
  if (j_name == nullptr || j_args == nullptr || j_flowdir == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  std::unordered_map<std::string, std::string> m_args;
  if (j_args != nullptr) {
    auto n_args =
        modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
            env, j_args);
    if (n_args == nullptr) {
      modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
      return;
    }

    for (const auto &key : n_args->GetKeys()) {
      m_args[key] = n_args->GetString(key);
    }
  }

  auto ret = n_flow->InitByName(modelbox::jstring2string(env, j_name), m_args,
                                modelbox::jstring2string(env, j_flowdir));
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowInit
 * Signature: (Ljava/lang/String;Lcom/modelbox/Configuration;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Flow_FlowInitByName__Ljava_lang_String_2Lcom_modelbox_Configuration_2(
    JNIEnv *env, jobject j_this, jstring j_name, jobject j_args) {
  if (j_name == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
  }

  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  std::unordered_map<std::string, std::string> m_args;
  if (j_args != nullptr) {
    auto n_args =
        modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
            env, j_args);
    if (n_args == nullptr) {
      modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
      return;
    }

    for (const auto &key : n_args->GetKeys()) {
      m_args[key] = n_args->GetString(key);
    }
  }

  auto ret = n_flow->InitByName(modelbox::jstring2string(env, j_name), m_args);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowRegisterFlowUnit
 * Signature: (Lcom/modelbox/FlowUnitBuilder;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Flow_FlowRegisterFlowUnit(
    JNIEnv *env, jobject j_this, jobject j_builder) {

  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  if (j_builder == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_builder =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitBuilder>(
          env, j_builder);
  if (n_builder == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_flow->RegisterFlowUnit(n_builder);
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowStop
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Flow_FlowStop(JNIEnv *env,
                                                       jobject j_this) {
  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_flow->Stop();
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowCreateExternalDataMap
 * Signature: ()Lcom/modelbox/ExternalDataMap;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_Flow_FlowCreateExternalDataMap(JNIEnv *env, jobject j_this) {
  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto datamap = n_flow->CreateExternalDataMap();
  if (datamap == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_FAULT,
                               "Create External data failed.");
    return nullptr;
  }

  jobject j_data_map = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/ExternalDataMap", datamap);
  if (j_data_map == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_data_map;
}

/*
 * Class:     com_modelbox_Flow
 * Method:    FlowCreateStreamIO
 * Signature: ()Lcom/modelbox/FlowStreamIO;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_Flow_FlowCreateStreamIO(JNIEnv *env, jobject j_this) {
  auto n_flow = modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Flow>(
      env, j_this);
  if (n_flow == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto stream_io = n_flow->CreateStreamIO();
  if (stream_io == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_FAULT,
                               "Create External data failed.");
    return nullptr;
  }

  jobject j_stream_io = modelbox::JNINativeObject::NewJObject(
      env, "com/modelbox/FlowStreamIO", stream_io);
  if (j_stream_io == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  return j_stream_io;
}
