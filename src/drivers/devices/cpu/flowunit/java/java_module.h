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

#ifndef MODELBOX_JAVA_FLOWUNIT_MODULE_H_
#define MODELBOX_JAVA_FLOWUNIT_MODULE_H_

#include <modelbox/base/status.h>
#include <jni.h>

class JavaJVM {
 public:
  JavaJVM();
  virtual ~JavaJVM();
  modelbox::Status InitJVM();
  modelbox::Status ExitJVM();

  JNIEnv *GetEnv();
 private:
  modelbox::Status InitJNI();
  modelbox::Status ExitJNI();
  bool is_initialized_ = false;
  bool is_jvm_create_ = false;
  JavaVM *jvm_ = nullptr;
  JNIEnv *env_;
};

extern std::shared_ptr<JavaJVM> kJavaJVM;

#endif  // MODELBOX_JAVA_FLOWUNIT_MODULE_H_