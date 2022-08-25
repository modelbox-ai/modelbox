
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

#include <memory>

#include "com_modelbox_FlowUnitEvent.h"
#include "jni_native_object.h"
#include "modelbox/data_context.h"
#include "scoped_jvm.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_FlowUnitEvent
 * Method:    FlowUnitEventNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_modelbox_FlowUnitEvent_FlowUnitEventNew(JNIEnv *env, jobject j_this) {
  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::FlowUnitEvent>());
}

/*
 * Class:     com_modelbox_FlowUnitEvent
 * Method:    FlowUnitEventSet
 * Signature: (Ljava/lang/String;Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_FlowUnitEvent_FlowUnitEventSet(
    JNIEnv *env, jobject j_this, jstring j_key, jobject j_object) {
  if (j_key == nullptr || j_object == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_flowunit_event =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitEvent>(
          env, j_this);
  if (n_flowunit_event == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto *j_global_object = env->NewGlobalRef(j_object);
  std::shared_ptr<void> priv_ptr(
      (void *)j_global_object, [](void *j_global_object) {
        modelbox::ScopedJvm scoped;
        scoped.GetJNIEnv()->DeleteGlobalRef((jobject)j_global_object);
      });
  n_flowunit_event->SetPrivate(modelbox::jstring2string(env, j_key), priv_ptr);
}

/*
 * Class:     com_modelbox_FlowUnitEvent
 * Method:    FlowUnitEventGet
 * Signature: (Ljava/lang/String;)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL Java_com_modelbox_FlowUnitEvent_FlowUnitEventGet(
    JNIEnv *env, jobject j_this, jstring j_key) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return nullptr;
  }

  auto n_flowunit_event =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::FlowUnitEvent>(
          env, j_this);
  if (n_flowunit_event == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_object =
      n_flowunit_event->GetPrivate(modelbox::jstring2string(env, j_key));
  if (n_object == nullptr) {
    return nullptr;
  }

  return (jobject)n_object.get();
}