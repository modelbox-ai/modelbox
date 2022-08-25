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

#include "com_modelbox_ExternalDataSelect.h"
#include "jni_native_object.h"
#include "modelbox/external_data_map.h"
#include "throw.h"
#include "utils.h"
#include "scoped_jvm.h"

/*
 * Class:     com_modelbox_ExternalDataSelect
 * Method:    ExternalDataSelect_New
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_modelbox_ExternalDataSelect_ExternalDataSelect_1New(JNIEnv *env,
                                                             jobject j_this) {
  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::ExternalDataSelect>());
}

/*
 * Class:     com_modelbox_ExternalDataSelect
 * Method:    ExternalDataSelect_RegisterExternalData
 * Signature: (Lcom/modelbox/ExternalDataMap;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_ExternalDataSelect_ExternalDataSelect_1RegisterExternalData(
    JNIEnv *env, jobject j_this, jobject j_data_map) {
  if (j_data_map == nullptr) {
    modelbox::ModelBoxJNIThrow(
        env, modelbox::STATUS_INVALID,
        "ExternalDataSelect Register: input argument is null");
    return;
  }

  auto n_data_select = modelbox::JNINativeObject::GetNativeSharedPtr<
      modelbox::ExternalDataSelect>(env, j_this);
  if (n_data_select == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_data_map =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_data_map);
  if (n_data_map == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto *j_global_data_map = env->NewGlobalRef(j_data_map);
  std::shared_ptr<void> priv_ptr(
      (void *)j_global_data_map, [](void *global_data_map) {
        modelbox::ScopedJvm scoped;
        scoped.GetJNIEnv()->DeleteGlobalRef((jobject)global_data_map);
      });
  n_data_map->SetPrivate(priv_ptr);
  n_data_select->RegisterExternalData(n_data_map);
}

/*
 * Class:     com_modelbox_ExternalDataSelect
 * Method:    ExternalDataSelect_RemoveExternalData
 * Signature: (Lcom/modelbox/ExternalDataMap;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_ExternalDataSelect_ExternalDataSelect_1RemoveExternalData(
    JNIEnv *env, jobject j_this, jobject j_data_map) {
  if (j_data_map == nullptr) {
    modelbox::ModelBoxJNIThrow(
        env, modelbox::STATUS_INVALID,
        "ExternalDataSelect Remove: input argument is null");
    return;
  }

  auto n_data_select = modelbox::JNINativeObject::GetNativeSharedPtr<
      modelbox::ExternalDataSelect>(env, j_this);
  if (n_data_select == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_data_map =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::ExternalDataMap>(
          env, j_data_map);
  if (n_data_map == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_data_select->RemoveExternalData(n_data_map);
}

/*
 * Class:     com_modelbox_ExternalDataSelect
 * Method:    ExternalDataSelect_SelectExternalData
 * Signature: (J)Ljava/util/ArrayList;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_ExternalDataSelect_ExternalDataSelect_1SelectExternalData(
    JNIEnv *env, jobject j_this, jlong j_timeout) {
  auto n_data_select = modelbox::JNINativeObject::GetNativeSharedPtr<
      modelbox::ExternalDataSelect>(env, j_this);
  if (n_data_select == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  std::list<std::shared_ptr<modelbox::ExternalDataMap>> datamap_list;
  auto ret = n_data_select->SelectExternalData(
      datamap_list, std::chrono::milliseconds((int64_t)j_timeout));
  if (ret != modelbox::STATUS_SUCCESS) {
    if (ret == modelbox::STATUS_TIMEDOUT) {
      return nullptr;
    }
    modelbox::ModelBoxJNIThrow(env, ret);
    return nullptr;
  }

  auto *j_arraylist_cls = env->FindClass("java/util/ArrayList");
  if (j_arraylist_cls == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_FAULT,
                               "cannot find array list");
    return nullptr;
  }
  Defer { env->DeleteLocalRef(j_arraylist_cls); };

  jmethodID j_list_add_ID =
      env->GetMethodID(j_arraylist_cls, "add", "(Ljava/lang/Object;)Z");
  jmethodID j_list_init_ID =
      env->GetMethodID(j_arraylist_cls, "<init>", "(I)V");

  if (j_list_add_ID == nullptr || j_list_init_ID == nullptr) {
    modelbox::ModelBoxJNIThrow(
        env, modelbox::STATUS_FAULT,
        "Cannot find arraylist functions add and <init>");
    return nullptr;
  }

  auto *j_arraylist =
      env->NewObject(j_arraylist_cls, j_list_init_ID, datamap_list.size());
  if (j_arraylist == nullptr) {
    ret = {modelbox::STATUS_NOMEM, "cannot create arraylist"};
    modelbox::ModelBoxJNIThrow(env, ret);
    return nullptr;
  }

  for (const auto &datamap : datamap_list) {
    std::shared_ptr<void> object = datamap->GetPrivate<void>();
    if (object == nullptr) {
      continue;
    }

    env->CallBooleanMethod(j_arraylist, j_list_add_ID, (jobject)object.get());
  }

  return j_arraylist;
}