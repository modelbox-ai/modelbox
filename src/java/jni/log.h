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

#ifndef MODELBOX_JAVA_LIB_LOG_H_
#define MODELBOX_JAVA_LIB_LOG_H_

#include <modelbox/base/log.h>
#include <jni.h>

namespace modelbox {

class LoggerJava : public Logger {
 public:
  LoggerJava();
  virtual ~LoggerJava();

  virtual void Print(LogLevel level, const char *file, int lineno,
                     const char *func, const char *msg);

  void RegJNICaller(JNIEnv *env, jobject logger);

  void UnReg();

  void SetLogLevel(LogLevel level);

  virtual LogLevel GetLogLevel();

 private:
  JNIEnv *env_;
  jobject logger_;
  jmethodID log_mid_; 
  LogLevel level_{LOG_OFF};
};

class LoggerJavaWapper {
 public:
  LoggerJavaWapper();
  virtual ~LoggerJavaWapper();

  void RegLogFunc(std::string pylog);

  void SetLogLevel(LogLevel level);

  const std::shared_ptr<Logger> GetLogger();

  void SetLogger(std::shared_ptr<Logger> logger);

  void PrintExt(LogLevel level, const char *file, int lineno, const char *func,
                const char *msg);
  void Print(LogLevel level, const char *msg);

 private:
  std::shared_ptr<LoggerJava> logger_java_ = std::make_shared<LoggerJava>();
};

}  // namespace modelbox

#endif  // MODELBOX_JAVA_LIB_LOG_H_
