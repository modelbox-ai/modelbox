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

/**
 * modelbox Status
 */
public class Status extends NativeObject {
  public Status() {
    setNativeHandle(StatusNew());
  }

  /**
   * constructor of status
   * @param code status code
   * @param msg  status message
   */
  public Status(StatusCode code, String msg) {
    setNativeHandle(StatusNew());
    StatusSetCode(code.ordinal());
    StatusSetErrorMsg(msg);
  }

  /**
   * constructor of status
   * @param other another status
   * @param msg  status message
   */
  public Status(Status other, String msg) {
    setNativeHandle(StatusNew());
    StatusWrap(other, other.Code().ordinal(), msg);
  }

  /**
   * make status to string
   * @return status in string
   */
  public String ToSting() {
    return StatusToSting();
  }

  /**
   * Get status code
   * @return status code
   */
  public StatusCode Code() {
    return StatusCode();
  }

  /**
   * Get status code in string
   * @return status code in string
   */
  public String StrCode() {
    return StatusStrCode();
  }

  /**
   * Set error message to status
   * @param errorMsg error message
   */
  public void SetErrorMsg(String errorMsg) {
    StatusSetErrorMsg(errorMsg);
  }

  /**
   * Get error message
   * @return error messsage
   */
  public String ErrorMsg() {
    return StatusErrorMsg();
  }

  /**
   * Get wrap error message
   * @return wrap error message
   */
  public String WrapErrormsgs() {
    return StatusWrapErrormsgs();
  }

  private native long StatusNew();

  private native void StatusSetCode(long code);

  private native void StatusWrap(Status status, long code, String msg);

  private native String StatusToSting();

  private native StatusCode StatusCode();

  private native String StatusStrCode();

  private native void StatusSetErrorMsg(String errorMsg);

  private native String StatusErrorMsg();

  private native String StatusWrapErrormsgs();

  @Override
  public boolean equals(Object o) {
    if (o == null) {
      return false;
    }

    if (o instanceof Status == false) {
      return false;
    }

    if (Code() == ((Status) o).Code()) {
      return true;
    }

    return false;
  }
}
