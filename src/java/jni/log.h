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

#include <jni.h>
#include <modelbox/base/log.h>

namespace modelbox {

class LoggerJava : public Logger {
 public:
  LoggerJava();
  ~LoggerJava() override;

  void Print(LogLevel level, const char *file, int lineno, const char *func,
             const char *msg) override;

  void RegJNICaller(JNIEnv *env, jobject logger);

  jobject GetJNICaller();

  void UnReg(JNIEnv *env);

  void SetLogLevel(LogLevel level) override;

  LogLevel GetLogLevel() override;

 private:
  jobject logger_{nullptr};
  jmethodID log_mid_{nullptr};
  LogLevel level_{LOG_OFF};
};

class LoggerJavaWapper {
 public:
  LoggerJavaWapper();
  virtual ~LoggerJavaWapper();

  void RegLogFunc(const std::string &pylog);

  void SetLogLevel(LogLevel level);

  std::shared_ptr<Logger> GetLogger();

  void SetLogger(const std::shared_ptr<Logger> &logger);

  void PrintExt(LogLevel level, const char *file, int lineno, const char *func,
                const char *msg);
  void Print(LogLevel level, const char *msg);

 private:
  std::shared_ptr<LoggerJava> logger_java_ = std::make_shared<LoggerJava>();
};

}  // namespace modelbox

#endif  // MODELBOX_JAVA_LIB_LOG_H_
