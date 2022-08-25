
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

#include "com_modelbox_DataMeta.h"
#include "jni_native_object.h"
#include "modelbox/external_data_map.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_DataMeta
 * Method:    DataMetaNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_DataMeta_DataMetaNew(JNIEnv *env,
                                                               jobject j_this) {
  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::DataMeta>());
}

/*
 * Class:     com_modelbox_DataMeta
 * Method:    DataMetaSet
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_DataMeta_DataMetaSet(JNIEnv *env,
                                                              jobject j_this,
                                                              jstring j_key,
                                                              jstring j_value) {
  if (j_key == nullptr || j_value == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_datameta =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataMeta>(env,
                                                                        j_this);
  if (n_datameta == nullptr || j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_datameta->SetMeta(
      modelbox::jstring2string(env, j_key),
      std::make_shared<std::string>(modelbox::jstring2string(env, j_value)));
}

/*
 * Class:     com_modelbox_DataMeta
 * Method:    DataMetaGetString
 * Signature: (Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_modelbox_DataMeta_DataMetaGetString(
    JNIEnv *env, jobject j_this, jstring j_key) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return nullptr;
  }

  auto n_datameta =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::DataMeta>(env,
                                                                        j_this);
  if (n_datameta == nullptr || j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto value = n_datameta->GetMeta(modelbox::jstring2string(env, j_key));
  if (value == nullptr) {
    return nullptr;
  }

  auto n_value = std::static_pointer_cast<std::string>(value); 
  if (value == nullptr) {
    return nullptr;
  }

  return env->NewStringUTF(n_value->c_str());
}
