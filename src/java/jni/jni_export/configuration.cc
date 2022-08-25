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

#include "modelbox/base/configuration.h"

#include <memory>

#include "com_modelbox_Configuration.h"
#include "jni_native_object.h"
#include "scoped_jvm.h"
#include "throw.h"
#include "utils.h"

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationGetBoolean
 * Signature: (Ljava/lang/String;Z)Z
 */
JNIEXPORT jboolean JNICALL
Java_com_modelbox_Configuration_ConfigurationGetBoolean(JNIEnv *env,
                                                        jobject j_this,
                                                        jstring j_key,
                                                        jboolean j_default) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return false;
  }
  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return false;
  }

  return (jboolean)n_config->GetBool(modelbox::jstring2string(env, j_key),
                                     (bool)j_default);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationGetInt
 * Signature: (Ljava/lang/String;I)I
 */
JNIEXPORT jint JNICALL Java_com_modelbox_Configuration_ConfigurationGetInt(
    JNIEnv *env, jobject j_this, jstring j_key, jint j_default) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return 0;
  }
  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  return (jint)n_config->GetInt32(modelbox::jstring2string(env, j_key),
                                  (jint)j_default);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationGetLong
 * Signature: (Ljava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_com_modelbox_Configuration_ConfigurationGetLong(
    JNIEnv *env, jobject j_this, jstring j_key, jlong j_default) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return 0;
  }
  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  return (jlong)n_config->GetInt64(modelbox::jstring2string(env, j_key),
                                   (jlong)j_default);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationGetString
 * Signature: (Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_modelbox_Configuration_ConfigurationGetString(JNIEnv *env,
                                                       jobject j_this,
                                                       jstring j_key,
                                                       jstring j_default) {
  std::string defaultValue;
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return nullptr;
  }

  if (j_default) {
    defaultValue = modelbox::jstring2string(env, j_default);
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto retvalue =
      n_config->GetString(modelbox::jstring2string(env, j_key), defaultValue);
  return env->NewStringUTF(retvalue.c_str());
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationGetFloat
 * Signature: (Ljava/lang/String;F)F
 */
JNIEXPORT jfloat JNICALL Java_com_modelbox_Configuration_ConfigurationGetFloat(
    JNIEnv *env, jobject j_this, jstring j_key, jfloat j_default) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return (jfloat)0;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return (jfloat)0;
  }

  return (jfloat)n_config->GetFloat(modelbox::jstring2string(env, j_key),
                                    (jfloat)j_default);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationGetDouble
 * Signature: (Ljava/lang/String;D)D
 */
JNIEXPORT jdouble JNICALL
Java_com_modelbox_Configuration_ConfigurationGetDouble(JNIEnv *env,
                                                       jobject j_this,
                                                       jstring j_key,
                                                       jdouble j_default) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return 0;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return 0;
  }

  return (jdouble)n_config->GetDouble(modelbox::jstring2string(env, j_key),
                                      (jdouble)j_default);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationSet
 * Signature: (Ljava/lang/String;Z)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Configuration_ConfigurationSet__Ljava_lang_String_2Z(
    JNIEnv *env, jobject j_this, jstring j_key, jboolean j_value) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_config->SetProperty<bool>(modelbox::jstring2string(env, j_key),
                              (bool)j_value);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationSet
 * Signature: (Ljava/lang/String;I)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Configuration_ConfigurationSet__Ljava_lang_String_2I(
    JNIEnv *env, jobject j_this, jstring j_key, jint j_value) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_config->SetProperty<int32_t>(modelbox::jstring2string(env, j_key),
                                 (int32_t)j_value);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationSet
 * Signature: (Ljava/lang/String;J)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Configuration_ConfigurationSet__Ljava_lang_String_2J(
    JNIEnv *env, jobject j_this, jstring j_key, jlong j_value) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_config->SetProperty<int64_t>(modelbox::jstring2string(env, j_key),
                                 (int64_t)j_value);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationSet
 * Signature: (Ljava/lang/String;F)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Configuration_ConfigurationSet__Ljava_lang_String_2F(
    JNIEnv *env, jobject j_this, jstring j_key, jfloat j_value) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_config->SetProperty<float>(modelbox::jstring2string(env, j_key),
                               (float)j_value);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationSet
 * Signature: (Ljava/lang/String;D)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Configuration_ConfigurationSet__Ljava_lang_String_2D(
    JNIEnv *env, jobject j_this, jstring j_key, jdouble j_value) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  n_config->SetProperty<double>(modelbox::jstring2string(env, j_key),
                                (double)j_value);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationSet
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Configuration_ConfigurationSet__Ljava_lang_String_2Ljava_lang_String_2(
    JNIEnv *env, jobject j_this, jstring j_key, jstring j_value) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto n_value = modelbox::jstring2string(env, j_value);

  n_config->SetProperty<std::string>(modelbox::jstring2string(env, j_key),
                                     n_value);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationGetStrings
 * Signature: (Ljava/lang/String;Ljava/util/ArrayList;)Ljava/util/ArrayList;
 */
JNIEXPORT jobject JNICALL
Java_com_modelbox_Configuration_ConfigurationGetStrings(JNIEnv *env,
                                                        jobject j_this,
                                                        jstring j_key,
                                                        jobject j_default) {
  if (j_key == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return nullptr;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return nullptr;
  }

  auto n_values = n_config->GetStrings(modelbox::jstring2string(env, j_key));
  if (n_values.empty()) {
    return j_default;
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
      env->NewObject(j_arraylist_cls, j_list_init_ID, n_values.size());
  if (j_arraylist == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_NOMEM,
                               "cannot create arraylist");
    return nullptr;
  }

  for (const auto &n_value : n_values) {
    auto *j_obj = env->NewStringUTF(n_value.c_str());
    env->CallBooleanMethod(j_arraylist, j_list_add_ID, j_obj);
    env->DeleteLocalRef(j_obj);
  }

  return j_arraylist;
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationSet
 * Signature: (Ljava/lang/String;Ljava/util/ArrayList;)V
 */
JNIEXPORT void JNICALL
Java_com_modelbox_Configuration_ConfigurationSet__Ljava_lang_String_2Ljava_util_ArrayList_2(
    JNIEnv *env, jobject j_this, jstring j_key, jobject j_array_string) {
  if (j_key == nullptr || j_array_string == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  auto n_config =
      modelbox::JNINativeObject::GetNativeSharedPtr<modelbox::Configuration>(
          env, j_this);
  if (n_config == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto *j_arraylist_cls = env->FindClass("java/util/ArrayList");
  if (j_arraylist_cls == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_FAULT,
                               "cannot find array list");
    return;
  }
  Defer { env->DeleteLocalRef(j_arraylist_cls); };

  jmethodID j_list_get_ID =
      env->GetMethodID(j_arraylist_cls, "get", "(I)Ljava/lang/Object;");
  jmethodID j_list_size_ID = env->GetMethodID(j_arraylist_cls, "size", "()I");

  if (j_list_get_ID == nullptr || j_list_size_ID == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_FAULT,
                               "Cannot find arraylist functions get and size");
    return;
  }

  jsize j_arraylist_size = env->CallIntMethod(j_array_string, j_list_size_ID);
  if (j_arraylist_size <= 0) {
    return;
  }

  std::vector<std::string> n_values;
  n_values.reserve(j_arraylist_size);
  for (int i = 0; i < j_arraylist_size; i++) {
    auto *j_obj =
        (jstring)env->CallObjectMethod(j_array_string, j_list_get_ID, i);
    Defer { env->DeleteLocalRef(j_obj); };

    auto n_value = modelbox::jstring2string(env, j_obj);
    n_values.emplace_back(n_value);
  }

  n_config->SetProperty<std::string>(modelbox::jstring2string(env, j_key),
                                     n_values);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationParser
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_Configuration_ConfigurationParser(
    JNIEnv *env, jobject j_this, jstring j_file) {
  if (j_file == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::STATUS_INVALID,
                               "input argument is null");
    return;
  }

  modelbox::ConfigurationBuilder builder;
  auto n_newconfig = builder.Build(modelbox::jstring2string(env, j_file));
  if (n_newconfig == nullptr) {
    modelbox::ModelBoxJNIThrow(env, modelbox::StatusError);
    return;
  }

  auto ret =
      modelbox::JNINativeObject::SetNativeSharedPtr(env, j_this, n_newconfig);
  modelbox::ModelBoxJNIThrow(env, ret);
}

/*
 * Class:     com_modelbox_Configuration
 * Method:    ConfigurationNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_modelbox_Configuration_ConfigurationNew(JNIEnv *env, jobject j_this) {
  return modelbox::JNINativeObject::NewHandle(
      j_this, std::make_shared<modelbox::Configuration>());
}
