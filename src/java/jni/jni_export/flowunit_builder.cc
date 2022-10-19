
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
#include <modelbox/flowunit_builder.h>

#include <memory>

#include "com_modelbox_FlowUnitBuilder.h"
#include "jni_native_object.h"
#include "scoped_jvm.h"
#include "throw.h"
#include "utils.h"

class JavaFlowUnitBuilder : public modelbox::FlowUnitBuilder {
 public:
  JavaFlowUnitBuilder(jobject j_builder) {
    modelbox::ScopedJvm scope;
    auto *env = scope.GetJNIEnv();
    if (env == nullptr) {
      return;
    }

    j_builder_ = env->NewGlobalRef(j_builder);
  }

  virtual ~JavaFlowUnitBuilder() {
    modelbox::ScopedJvm scope;
    auto *env = scope.GetJNIEnv();
    if (env == nullptr) {
      return;
    }

    env->DeleteGlobalRef(j_builder_);
  }

  void Probe(std::shared_ptr<modelbox::FlowUnitDesc> &desc) override {
    modelbox::ScopedJvm scope;
    auto *env = scope.GetJNIEnv();
    if (env == nullptr) {
      MBLOG_ERROR << "Failed to get JNIEnv";
      return;
    }

    auto *j_desc = modelbox::JNINativeObject::NewJObject(
        env, "com/modelbox/FlowUnitDesc", desc);
    if (j_desc == nullptr) {
      MBLOG_ERROR << "new java flowunit desc failed" << modelbox::StatusError;
      return;
    }
    Defer { env->DeleteLocalRef(j_desc); };

    jmethodID probe_method =
        env->GetMethodID(env->GetObjectClass(j_builder_), "probe",
                         "(Lcom/modelbox/FlowUnitDesc;)V");
    if (probe_method == nullptr) {
      MBLOG_ERROR << "get probe method failed.";
      return;
    }

    env->CallVoidMethod(j_builder_, probe_method, j_desc);
    if (env->ExceptionCheck() == JNI_TRUE) {
      std::string java_stack;
      std::string errmsg = modelbox::ModelboxExceptionMsg(env, &java_stack);
      MBLOG_WARN << "flowunit probe exception: " << errmsg << "\n"
                 << java_stack;
    }
  }

  std::shared_ptr<modelbox::FlowUnit> Build() override {
    modelbox::ScopedJvm scope;
    auto *env = scope.GetJNIEnv();
    if (env == nullptr) {
      return nullptr;
    }

    jmethodID build_method = env->GetMethodID(
        env->GetObjectClass(j_builder_), "build", "()Lcom/modelbox/FlowUnit;");
    if (build_method == nullptr) {
      MBLOG_ERROR << "get build method failed.";
      return nullptr;
    }

    jobject j_flow_unit = env->CallObjectMethod(j_builder_, build_method);
    if (j_flow_unit == nullptr) {
      if (env->ExceptionCheck() == JNI_TRUE) {
        std::string java_stack;
        std::string errmsg = modelbox::ModelboxExceptionMsg(env, &java_stack);
        MBLOG_WARN << "flowunit builder exception: " << errmsg << "\n"
                   << java_stack;
        modelbox::StatusError = {modelbox::STATUS_FAULT,
                                 "flowunit builder exception:" + errmsg};
      }
      modelbox::StatusError = {modelbox::STATUS_FAULT,
                               "flowunit builder failed"};
      return nullptr;
    }

    auto n_flowunit =
        modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnit>(
            env, j_flow_unit);
    if (n_flowunit == nullptr) {
      MBLOG_ERROR << "get native flowunit failed" << modelbox::StatusError;
      modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
      return nullptr;
    }

    return n_flowunit;
  }

 private:
  jobject j_builder_;
};

/*
 * Class:     com_modelbox_FlowUnitBuilder
 * Method:    FlowUnitBuilderNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_FlowUnitBuilder_FlowUnitBuilderNew(
    JNIEnv *env, jobject j_this) {
  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<JavaFlowUnitBuilder>(j_this));
}
