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
   * Set long to meta data
   * @key meta key
   * @value meta value
   */ 
  public void setMetaLong(String key, long value) throws ModelBoxException {
    BufferSetMetaLong(key, value);
  }

  /**
   * Set int to meta data
   * @key meta key
   * @value meta value
   */
  public void setMetaInt(String key, int value) throws ModelBoxException {
    BufferSetMetaInt(key, value);
  }

  /**
   * Set String to meta data
   * @key meta key
   * @value meta value
   */
  public void setMetaString(String key, String value) throws ModelBoxException {
    BufferSetMetaString(key, value);
  }

  /**
   * Get double from meta data
   * @key meta key
   * @return meta value
   */
  public void setMetaDouble(String key, double value) throws ModelBoxException {
    BufferSetMetaDouble(key, value);
  }

  /**
   * Get float from meta data
   * @key meta key
   * @return meta value
   */
  public void setMetaFloat(String key, float value) throws ModelBoxException {
    BufferSetMetaFloat(key, value);
  }

  /**
   * Get boolean from meta data
   * @key meta key
   * @return meta value
   */
  public void setMetaBoolean(String key, boolean value) throws ModelBoxException {
    BufferSetMetaBoolean(key, value);
  }

  /**
   * Get long from meta data
   * @key meta key
   * @return meta value
   */
  public long getMetaLong(String key) throws ModelBoxException {
    return BufferGetMetaLong(key);
  }

  /**
   * Get int from meta data
   * @key meta key
   * @default default value
   * @return meta value
   */
  public long getMetaLong(String key, long defaultValue) {
    try {
      return getMetaLong(key);
    } catch (ModelBoxException e) {
      return defaultValue;
    }
  }

  /**
   * Get int from meta data
   * @key meta key
   * @return meta value
   */
  public int getMetaInt(String key) throws ModelBoxException {
    return BufferGetMetaInt(key);
  }

  /**
   * Get int from meta data
   * @key meta key
   * @default default value
   * @return meta value
   */
  public int getMetaInt(String key, int defaultValue) {
    try {
      return getMetaInt(key);
    } catch (ModelBoxException e) {
      return defaultValue;
    }
  }

  /**
   * Get String from meta data
   * @key meta key
   * @return meta value
   */
  public String getMetaString(String key) throws ModelBoxException {
    return BufferGetMetaString(key);
  }

  /**
   * Get String from meta data
   * @key meta key
   * @default default value
   * @return meta value
   */
  public String getMetaString(String key, String defaultValue) {
    try {
      return getMetaString(key);
    } catch (ModelBoxException e) {
      return defaultValue;
    }
  }

  /**
   * Get double from meta data
   * @key meta key
   * @return meta value
   */
  public double getMetaDouble(String key) throws ModelBoxException {
    return BufferGetMetaDouble(key);
  }

  /**
   * Get double from meta data
   * @key meta key
   * @default default value
   * @return meta value
   */
  public double getMetaDouble(String key, double defaultValue) {
    try {
      return getMetaDouble(key);
    } catch (ModelBoxException e) {
      return defaultValue;
    }
  }

  /**
   * Get double from meta data
   * @key meta key
   * @return meta value
   */
  public float getMetaFloat(String key) throws ModelBoxException {
    return BufferGetMetaFloat(key);
  }

  /**
   * Get float from meta data
   * @key meta key
   * @default default value
   * @return meta value
   */
  public float getMetaFloat(String key, float defaultValue) {
    try {
      return getMetaFloat(key);
    } catch (ModelBoxException e) {
      return defaultValue;
    }
  }

  /**
   * Get boolean from meta data
   * @key meta key
   * @return meta value
   */
  public boolean getMetaBool(String key) throws ModelBoxException {
    return BufferGetMetaBool(key);
  }

  /**
   * Get boolean from meta data
   * @key meta key
   * @default default value
   * @return meta value
   */
  public boolean getMetaBool(String key, boolean defaultValue) {
    try {
      return getMetaBool(key);
    } catch (ModelBoxException e) {
      return defaultValue;
    }
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

  private native void BufferSetMetaLong(String key, long value);

  private native void BufferSetMetaInt(String key, int value);

  private native void BufferSetMetaString(String key, String value);

  private native void BufferSetMetaDouble(String key, double value);

  private native void BufferSetMetaFloat(String key, float value);

  private native void BufferSetMetaBoolean(String key, boolean value);

  private native long BufferGetMetaLong(String key);

  private native int BufferGetMetaInt(String key);

  private native String BufferGetMetaString(String key);

  private native double BufferGetMetaDouble(String key);

  private native float BufferGetMetaFloat(String key);

  private native boolean BufferGetMetaBool(String key);

  private native void BufferCopyMeta(Buffer buffer, boolean isOverWrite);

  private native Device BufferGetDevice();
}
