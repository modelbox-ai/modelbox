
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

#include "scoped_jvm.h"
#include "throw.h"

namespace modelbox {

LoggerJava::LoggerJava() = default;
LoggerJava::~LoggerJava() {
  modelbox::ScopedJvm scope;
  UnReg(scope.GetJNIEnv());
}

void LoggerJava::Print(LogLevel level, const char *file, int lineno,
                       const char *func, const char *msg) {
  modelbox::ScopedJvm scope;
  auto *env = scope.GetJNIEnv();
  if (env == nullptr) {
    return;
  }
  auto *jfile = env->NewStringUTF(file);
  auto jlineno = (jint)lineno;
  auto *jfunc = env->NewStringUTF(func);
  auto *jmsg = env->NewStringUTF(msg);
  env->CallObjectMethod(logger_, log_mid_, (jlong)level, jfile, jlineno, jfunc,
                        jmsg);
  env->DeleteLocalRef(jfile);
  env->DeleteLocalRef(jfunc);
  env->DeleteLocalRef(jmsg);
}

jobject LoggerJava::GetJNICaller() { return logger_; }

void LoggerJava::RegJNICaller(JNIEnv *env, jobject logger) {
  UnReg(env);
  jclass cls = env->GetObjectClass(logger);
  if (cls == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "logger class is not found.");
    return;
  }

  Defer { env->DeleteLocalRef(cls); };

  jmethodID mid = env->GetMethodID(
      cls, "jniPrintCallback",
      "(JLjava/lang/String;ILjava/lang/String;Ljava/lang/String;)V");
  if (mid == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "no print callback function found.");
    return;
  }

  logger_ = env->NewGlobalRef(logger);
  log_mid_ = mid;
}

void LoggerJava::UnReg(JNIEnv *env) {
  if (logger_ == nullptr) {
    return;
  }

  env->DeleteLocalRef(logger_);
  logger_ = nullptr;
  log_mid_ = nullptr;
}

void LoggerJava::SetLogLevel(LogLevel level) { level_ = level; }

LogLevel LoggerJava::GetLogLevel() { return level_; }

LoggerJavaWapper::LoggerJavaWapper() = default;

LoggerJavaWapper::~LoggerJavaWapper() { ModelBoxLogger.SetLogger(nullptr); }

void LoggerJavaWapper::RegLogFunc(const std::string &pylog) {
  ModelBoxLogger.SetLogger(logger_java_);
}

std::shared_ptr<Logger> LoggerJavaWapper::GetLogger() {
  return ModelBoxLogger.GetLogger();
}

void LoggerJavaWapper::SetLogger(const std::shared_ptr<Logger> &logger) {
  ModelBoxLogger.SetLogger(logger);
}

void LoggerJavaWapper::SetLogLevel(LogLevel level) {
  logger_java_->SetLogLevel(level);
}

void LoggerJavaWapper::PrintExt(LogLevel level, const char *file, int lineno,
                                const char *func, const char *msg) {
  ModelBoxLogger.Print(level, file, lineno, func, "%s", msg);
}

}  // namespace modelbox
