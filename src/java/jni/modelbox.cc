
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

#include "modelbox.h"

#include <modelbox/base/log.h>
#include <modelbox/flow.h>

#include <memory>

#include "com_modelbox_ModelBoxJni.h"
#include "log.h"
#include "utils.h"

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    FlowNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_ModelBoxJni_FlowNew(JNIEnv *env,
                                                              jclass clazz) {
  std::shared_ptr<modelbox::Flow> *pflow = new std::shared_ptr<modelbox::Flow>;
  *pflow = std::make_shared<modelbox::Flow>();
  return reinterpret_cast<jlong>(pflow);
}

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    FlowFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBoxJni_FlowFree(JNIEnv *env,
                                                              jclass clazz,
                                                              jlong flow) {
  std::shared_ptr<modelbox::Flow> *pflow =
      reinterpret_cast<std::shared_ptr<modelbox::Flow> *>(flow);
  delete pflow;
}

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    FlowInit
 * Signature: (JLjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBoxJni_FlowInit(
    JNIEnv *env, jclass clazz, jlong flow, jstring jname, jstring jgraph) {
  auto name = jstring2string(env, jname);
  auto graph = jstring2string(env, jgraph);

  std::shared_ptr<modelbox::Flow> *pflow =
      reinterpret_cast<std::shared_ptr<modelbox::Flow> *>(flow);
  auto ret = (*pflow)->Init(name, graph);
}

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    FlowBuild
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBoxJni_FlowBuild(JNIEnv *env,
                                                               jclass clazz,
                                                               jlong flow) {
  std::shared_ptr<modelbox::Flow> *pflow =
      reinterpret_cast<std::shared_ptr<modelbox::Flow> *>(flow);
  auto ret = (*pflow)->Build();
}

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    LogNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_ModelBoxJni_LogNew(JNIEnv *env,
                                                             jclass clazz) {
  std::shared_ptr<modelbox::LoggerJava> *plog =
      new std::shared_ptr<modelbox::LoggerJava>;
  *plog = std::make_shared<modelbox::LoggerJava>();
  return reinterpret_cast<jlong>(plog);
}

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    LogFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBoxJni_LogFree(JNIEnv *env,
                                                             jclass clazz,
                                                             jlong jlog) {
  std::shared_ptr<modelbox::LoggerJava> *plog =
      reinterpret_cast<std::shared_ptr<modelbox::LoggerJava> *>(jlog);
  delete plog;
}

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    SetLogLevel
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBoxJni_SetLogLevel(JNIEnv *env,
                                                                 jclass clazz,
                                                                 jlong jlog,
                                                                 jlong jlevel) {
  std::shared_ptr<modelbox::LoggerJava> *plog =
      reinterpret_cast<std::shared_ptr<modelbox::LoggerJava> *>(jlog);
  (*plog)->SetLogLevel((modelbox::LogLevel)jlevel);
}

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    LogReg
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBoxJni_LogReg(JNIEnv *env,
                                                            jclass clazz,
                                                            jobject jlog) {
  jclass cls = env->GetObjectClass(jlog);
  jmethodID mid = env->GetMethodID(cls, "getLogPtr", "()J");
  jobject result = env->CallObjectMethod(jlog, mid);
  std::shared_ptr<modelbox::LoggerJava> *plog =
      reinterpret_cast<std::shared_ptr<modelbox::LoggerJava> *>(result);

  (*plog)->RegJNICaller(env, jlog);
  ModelBoxLogger.SetLogger(*plog);
}

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    LogUnReg
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBoxJni_LogUnReg(JNIEnv *env,
                                                              jclass clazz) {
  ModelBoxLogger.SetLogger(nullptr);
                                                              }

/*
 * Class:     com_modelbox_ModelBoxJni
 * Method:    LogPrint
 * Signature: (JLjava/lang/String;ILjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_ModelBoxJni_LogPrint(
    JNIEnv *env, jclass clazz, jlong jlevel, jstring jfile, jint jlineno,
    jstring jfunc, jstring jmsg) {
  ModelBoxLogger.Print((modelbox::LogLevel)jlevel,
                       jstring2string(env, jfile).c_str(), jlineno,
                       jstring2string(env, jfunc).c_str(), "%s",
                       jstring2string(env, jmsg).c_str());
}