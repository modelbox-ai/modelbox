
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

#include "log.h"

#include <modelbox/base/log.h>

#include <memory>

#include "com_modelbox_Log.h"
#include "jni_native_object.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_Log
 * Method:    LogNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_Log_LogNew(JNIEnv *env,
                                                     jobject j_this) {
  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::LoggerJava>());
}

/*
 * Class:     com_modelbox_Log
 * Method:    LogSetLogLevel
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Log_LogSetLogLevel(JNIEnv *env,
                                                            jobject j_this,
                                                            jlong j_level) {
  auto n_log =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::LoggerJava>(
          env, j_this);
  if (n_log == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_log->SetLogLevel((modelbox::LogLevel)j_level);
}

/*
 * Class:     com_modelbox_Log
 * Method:    LogGetLogLevel
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_Log_LogGetLogLevel(JNIEnv *env,
                                                             jobject j_this) {
  auto n_log =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::LoggerJava>(
          env, j_this);
  if (n_log == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return (jlong)modelbox::LogLevel::LOG_OFF;
  }

  auto logLevel = n_log->GetLogLevel();
  if (logLevel > modelbox::LogLevel::LOG_OFF) {
    return (jlong)modelbox::LogLevel::LOG_OFF;
  }
  return (jlong)logLevel;
}

/*
 * Class:     com_modelbox_Log
 * Method:    LogGetLogger
 * Signature: ()Lcom/modelbox/Log;
 */
JNIEXPORT jobject JNICALL Java_com_modelbox_Log_LogGetLogger(JNIEnv *env,
                                                             jclass j_clazz) {
  auto n_log = ModelBoxLogger.GetLogger();
  if (n_log == nullptr) {
    return nullptr;
  }

  auto logger_java = std::dynamic_pointer_cast<modelbox::LoggerJava>(n_log);
  if (logger_java == nullptr) {
    jobject j_log =
        modelbox::JNINativeObject::NewJObject(env, "com/modelbox/Log", n_log);
    return j_log;
  }

  auto *j_log = logger_java->GetJNICaller();

  return j_log;
}

/*
 * Class:     com_modelbox_Log
 * Method:    LogReg
 * Signature: (Lcom/modelbox/Log;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Log_LogReg(JNIEnv *env, jclass j_clazz,
                                                    jobject j_log) {
  auto n_log =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::LoggerJava>(
          env, j_log);
  if (n_log == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_log->RegJNICaller(env, j_log);
  ModelBoxLogger.SetLogger(n_log);
}

/*
 * Class:     com_modelbox_Log
 * Method:    LogUnReg
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Log_LogUnReg(JNIEnv *env,
                                                      jclass j_clazz) {
  ModelBoxLogger.SetLogger(nullptr);
}

/*
 * Class:     com_modelbox_Log
 * Method:    LogPrint
 * Signature: (JLjava/lang/String;ILjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Log_LogPrint(
    JNIEnv *env, jclass j_clazz, jlong j_level, jstring j_file, jint j_lineno,
    jstring j_func, jstring j_msg) {
  ModelBoxLogger.Print((modelbox::LogLevel)j_level,
                       modelbox::jstring2string(env, j_file).c_str(), j_lineno,
                       modelbox::jstring2string(env, j_func).c_str(), "%s",
                       modelbox::jstring2string(env, j_msg).c_str());
}
