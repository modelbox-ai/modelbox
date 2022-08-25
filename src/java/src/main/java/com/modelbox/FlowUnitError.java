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

public class FlowUnitError extends NativeObject {
  public FlowUnitError(String desc) {
    setNativeHandle(FlowUnitError_New(desc));
  }

  public FlowUnitError(String node, String error_pos, Status error_status) {
    setNativeHandle(FlowUnitError_New(node, error_pos, error_status));
  }

  /**
   * Get flowunit error description
   * @return error description
   */
  public String getDesc() {
    return FlowUnitError_GetDesc();
  }

  /**
   * Get flowunit error status
   * @return
   */
  public Status GetStatus() {
    return FlowUnitError_GetStatus();
  }

  private native long FlowUnitError_New(String desc);

  private native long FlowUnitError_New(String node, String error_pos, Status error_status);

  private native String FlowUnitError_GetDesc();

  private native Status FlowUnitError_GetStatus();
}
