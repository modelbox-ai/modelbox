
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

#include <memory>

#include "com_modelbox_NativeObject.h"
#include "jni_native_object.h"
#include "throw.h"

/*
 * Class:     com_modelbox_NativeObject
 * Method:    delete_handle
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_modelbox_NativeObject_delete_1handle(
    JNIEnv *env, jobject jself, jlong handle) {
  modelbox::JNINativeObject::DeleteHandle(handle);
}
