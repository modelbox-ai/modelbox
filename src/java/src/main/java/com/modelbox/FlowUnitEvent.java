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

package com.modelbox;

public class FlowUnitEvent extends NativeObject {

  FlowUnitEvent() {
    setNativeHandle(FlowUnitEventNew());
  }

  /**
   * set data
   * @param key data key
   * @param value data value
   */
  public void set(String key, String value) {
    FlowUnitEventSet(key, value);
  }

  /**
   * set data
   * @param key data key
   * @param value data value
   */
  public void set(String key, Object object) {
    FlowUnitEventSet(key, object);
  }


  public Object get(String key) {
    return FlowUnitEventGet(key);
  }

  private native long FlowUnitEventNew();

  private native void FlowUnitEventSet(String key, Object object);

  private native Object FlowUnitEventGet(String key);
}
