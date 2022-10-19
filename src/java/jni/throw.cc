

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

#include "throw.h"

#include "modelbox/base/log.h"
#include "modelbox/base/status.h"
#include "modelbox_jni.h"
#include "utils.h"

namespace modelbox {

const char *kModelBoxExceptionCodeMap[] = {
    "com/modelbox/ModelBoxException$Success",
    "com/modelbox/ModelBoxException$Fault",
    "com/modelbox/ModelBoxException$Notfound",
    "com/modelbox/ModelBoxException$Invalid",
    "com/modelbox/ModelBoxException$Again",
    "com/modelbox/ModelBoxException$Badconf",
    "com/modelbox/ModelBoxException$Nomem",
    "com/modelbox/ModelBoxException$Range",
    "com/modelbox/ModelBoxException$Exist",
    "com/modelbox/ModelBoxException$Internal",
    "com/modelbox/ModelBoxException$Busy",
    "com/modelbox/ModelBoxException$Permit",
    "com/modelbox/ModelBoxException$Notsupport",
    "com/modelbox/ModelBoxException$Nodata",
    "com/modelbox/ModelBoxException$Nospace",
    "com/modelbox/ModelBoxException$Nobufs",
    "com/modelbox/ModelBoxException$Overflow",
    "com/modelbox/ModelBoxException$Inprogress",
    "com/modelbox/ModelBoxException$Already",
    "com/modelbox/ModelBoxException$Timedout",
    "com/modelbox/ModelBoxException$Nostream",
    "com/modelbox/ModelBoxException$Reset",
    "com/modelbox/ModelBoxException$Continue",
    "com/modelbox/ModelBoxException$Edquot",
    "com/modelbox/ModelBoxException$Stop",
    "com/modelbox/ModelBoxException$Shutdown",
    "com/modelbox/ModelBoxException$Eof",
    "com/modelbox/ModelBoxException$Noent",
    "com/modelbox/ModelBoxException$Deadlock",
    "com/modelbox/ModelBoxException$Noresponse",
    "com/modelbox/ModelBoxException$Io",
};

static void do_jni_throw(JNIEnv *env, const char *except_name,
                         const char *message) {
  jclass ecls = env->FindClass(except_name);

  if (ecls == nullptr) {
    ecls = env->FindClass("java/lang/RuntimeException");
    message = "Modelbox: cannot find exception";
    if (ecls == nullptr) {
      MBLOG_ERROR << "Modelbox-JNI: Failed to throw exception";
      return;
    }
  }

  Defer { env->DeleteLocalRef(ecls); };
  int ret = env->ThrowNew(ecls, message);
  if (ret < 0) {
    MBLOG_ERROR << "Modelbox-JNI: Fatal Error";
  }
}

void ModelBoxJNIThrow(JNIEnv *env, Status &status) {
  ModelBoxJNIThrow(env, status.Code(), status.WrapErrormsgs());
}

void ModelBoxJNIThrow(JNIEnv *env, StatusCode code,
                      const std::string &errormsg) {
  if (code == STATUS_OK) {
    return;
  }

  if (code >= sizeof(kModelBoxExceptionCodeMap) / sizeof(char *)) {
    do_jni_throw(env, "java/lang/RuntimeException",
                 "Modelbox: Status is invalid.");
    return;
  }

  do_jni_throw(env, kModelBoxExceptionCodeMap[code], errormsg.c_str());
}

void ModelBoxJNIThrow(JNIEnv *env, const char *runtime_exception,
                      const char *errmsg) {
  do_jni_throw(env, runtime_exception, errmsg);
}

std::shared_ptr<Status> ModelboxJNICatchException(JNIEnv *env) {
  auto status = std::make_shared<Status>();
  if (env->ExceptionCheck() == JNI_FALSE) {
    return status;
  }

  auto *j_throw = env->ExceptionOccurred();
  if (j_throw == nullptr) {
    return nullptr;
  }
  Defer { env->DeleteLocalRef(j_throw); };

  jclass throwable_class = env->FindClass("java/lang/Throwable");
  if (throwable_class == nullptr) {
    return nullptr;
  }
  Defer { env->DeleteLocalRef(throwable_class); };

  for (size_t i = 0; i < sizeof(kModelBoxExceptionCodeMap) / sizeof(char *);
       i++) {
    auto *j_cls = env->FindClass(kModelBoxExceptionCodeMap[i]);
    if (j_cls == nullptr) {
      continue;
    }
    Defer { env->DeleteLocalRef(j_cls); };

    if (env->IsInstanceOf(j_throw, j_cls)) {
      jmethodID get_message = env->GetMethodID(throwable_class, "getMessage",
                                               "()Ljava/lang/String;");
      if (get_message == nullptr) {
        return nullptr;
      }

      auto *j_message = (jstring)env->CallObjectMethod(j_throw, get_message);
      if (j_message == nullptr) {
        return nullptr;
      }
      Defer { env->DeleteLocalRef(j_message); };
      auto msg = modelbox::jstring2string(env, j_message);
      *status = {static_cast<StatusCode>(i), msg};
      env->ExceptionClear();
      return status;
    }
  }

  return nullptr;
}

std::string ModelboxExceptionMsg(JNIEnv *env, std::string *stack) {
  std::string msg;
  auto *j_throw = env->ExceptionOccurred();
  if (j_throw == nullptr) {
    return "";
  }
  Defer { env->DeleteLocalRef(j_throw); };
  env->ExceptionClear();

  jclass throwable_class = env->FindClass("java/lang/Throwable");
  if (throwable_class == nullptr) {
    return "";
  }
  Defer { env->DeleteLocalRef(throwable_class); };

  jmethodID get_message =
      env->GetMethodID(throwable_class, "getMessage", "()Ljava/lang/String;");
  if (get_message == nullptr) {
    return "";
  }

  auto *j_message = (jstring)env->CallObjectMethod(j_throw, get_message);
  if (j_message == nullptr) {
    return "";
  }
  Defer { env->DeleteLocalRef(j_message); };
  msg = modelbox::jstring2string(env, j_message);

  if (stack == nullptr) {
    return msg;
  }

  /**
   * get stack
   */
  jmethodID get_stack = env->GetMethodID(throwable_class, "getStackTrace",
                                         "()[Ljava/lang/StackTraceElement;");
  if (get_stack == nullptr) {
    return msg;
  }

  auto *j_stack = (jobjectArray)env->CallObjectMethod(j_throw, get_stack);
  if (j_stack == nullptr) {
    return msg;
  }
  Defer { env->DeleteLocalRef(j_stack); };

  jclass stack_element_class = env->FindClass("java/lang/StackTraceElement");
  if (stack_element_class == nullptr) {
    return msg;
  }
  Defer { env->DeleteLocalRef(stack_element_class); };

  jmethodID get_class_name = env->GetMethodID(
      stack_element_class, "getClassName", "()Ljava/lang/String;");
  jmethodID get_method_name = env->GetMethodID(
      stack_element_class, "getMethodName", "()Ljava/lang/String;");
  jmethodID get_file_name = env->GetMethodID(stack_element_class, "getFileName",
                                             "()Ljava/lang/String;");
  jmethodID get_line_number =
      env->GetMethodID(stack_element_class, "getLineNumber", "()I");
  if (get_class_name == nullptr || get_method_name == nullptr ||
      get_file_name == nullptr || get_line_number == nullptr) {
    return msg;
  }

  jsize len = env->GetArrayLength(j_stack);
  for (int i = 0; i < len; i++) {
    auto *j_element = env->GetObjectArrayElement(j_stack, i);
    if (j_element == nullptr) {
      continue;
    }
    Defer { env->DeleteLocalRef(j_element); };

    auto *j_class_name =
        (jstring)env->CallObjectMethod(j_element, get_class_name);
    auto *j_method_name =
        (jstring)env->CallObjectMethod(j_element, get_method_name);
    auto *j_file_name =
        (jstring)env->CallObjectMethod(j_element, get_file_name);
    int j_line_number = (jlong)env->CallIntMethod(j_element, get_line_number);
    if (j_class_name == nullptr || j_method_name == nullptr ||
        j_file_name == nullptr) {
      continue;
    }

    *stack += modelbox::jstring2string(env, j_class_name) + "." +
              modelbox::jstring2string(env, j_method_name) + "(" +
              modelbox::jstring2string(env, j_file_name) + ":";
    if (j_line_number < 0) {
      *stack += "jni";
    } else {
      *stack += std::to_string(j_line_number);
    }
    *stack += ")\n";
    env->DeleteLocalRef(j_class_name);
    env->DeleteLocalRef(j_method_name);
    env->DeleteLocalRef(j_file_name);
  }

  return msg;
}

}  // namespace modelbox