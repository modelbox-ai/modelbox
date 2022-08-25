

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

std::string ModelboxExceptionMsg(JNIEnv *env) {
  auto *j_throw = env->ExceptionOccurred();
  if (j_throw == nullptr) {
    return "";
  }

  // TODO
  return "";
}

}  // namespace modelbox