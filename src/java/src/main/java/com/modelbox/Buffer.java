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
 * Modelbox Buffer
 */
public class Buffer extends NativeObject {
  private Buffer() {

  }

  /**
   * Build buffer by size
   * @param size
   */
  public void build(long size) throws ModelBoxException {
    BufferBuild(size);
  }

  /**
   * build buffer from bytes[]
   * @param data
   */
  public void build(byte[] data) throws ModelBoxException {
    BufferBuild(data);
  }

  /**
   * get bytes[] from buffer
   * @return
   */
  public byte[] getData() throws ModelBoxException {
    return BufferGetData();
  }

  /*
   * Any error on buffer
   */
  public boolean hasError() {
    return BufferHasError();
  }

  /**
   * Set error to buffer
   * @param code error code
   * @param message error message
   */
  public void setError(String code, String message) throws ModelBoxException {
    BufferSetError(code, message);
  }

  /**
   * Get error code
   * @return error code
   */
  public String getErrorCode() {
    return BufferGetErrorCode();
  }

  /**
   * Get error message
   * @return error message
   */
  public String getErrorMsg() {
    return BufferGetErrorMsg();
  }

  /**
   * Get buffer length in byte
   * @return
   */
  public long getBytes() {
    return BufferGetBytes();
  }

  /**
   * Copy meta from another buffer
   * @param buffer another buffer
   * @param isOverWrite overwrite exist meta
   * @throws ModelBoxException
   */
  public void copyMeta(Buffer buffer, boolean isOverWrite) throws ModelBoxException {
    BufferCopyMeta(buffer, isOverWrite);
  }

  /**
   * Copy meta from anoter buffer
   * @param buffer another meta
   */
  public void copyMeta(Buffer buffer) {
    BufferCopyMeta(buffer, false);
  }

  /**
   * Get buffer device
   * @return
   */
  public Device getDevice() {
    return BufferGetDevice();
  }

  private native void BufferBuild(long size);

  private native void BufferBuild(byte[] data);

  private native byte[] BufferGetData();

  private native boolean BufferHasError();

  private native void BufferSetError(String code, String message);

  private native String BufferGetErrorCode();

  private native String BufferGetErrorMsg();

  private native long BufferGetBytes();

  private native void BufferCopyMeta(Buffer buffer, boolean isOverWrite);

  private native Device BufferGetDevice();
}
