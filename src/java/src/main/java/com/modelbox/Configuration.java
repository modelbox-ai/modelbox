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

import java.util.ArrayList;

/**
 * modelbox Configuration
 */
public class Configuration extends NativeObject {
  public Configuration() {
    setNativeHandle(ConfigurationNew());
  }

  /**
   * Parser configuration from file
   * @param file toml or json file
   * @throws ModelBoxException
   */
  public void Parser(String file) throws ModelBoxException {
    ConfigurationParser(file);
  }

  /**
   * Get boolean key
   * @param key
   * @return
   */
  public boolean getBoolean(String key, boolean defaultValue) {
    return ConfigurationGetBoolean(key, defaultValue);
  }

  /**
   * Get int key
   * @param key
   * @return
   */
  public int getInt(String key, int defaultValue) {
    return ConfigurationGetInt(key, defaultValue);
  }

  /**
   * Get long key
   * @param key
   * @return
   */
  public long getLong(String key, long defaultValue) {
    return ConfigurationGetLong(key, defaultValue);
  }

  /**
   * Get String key
   * @param key
   * @return
   */
  public String getString(String key, String defaultValue) {
    return ConfigurationGetString(key, defaultValue);
  }

  /**
   * Get float key
   * @param key
   * @return
   */
  public float getFloat(String key, float defaultValue) {
    return ConfigurationGetFloat(key, defaultValue);
  }

  /**
   * Get double key
   * @param key
   * @return
   */
  public double getDouble(String key, double defaultValue) {
    return ConfigurationGetDouble(key, defaultValue);
  }

  /**
   * Set boolean key
   * @param key
   * @param value
   */
  public void set(String key, boolean value) {
    ConfigurationSet(key, value);
  }

  /**
   * Set int key
   */
  public void set(String key, int value) {
    ConfigurationSet(key, value);
  }

  /**
   * Set long key
   */
  public void set(String key, long value) {
    ConfigurationSet(key, value);
  }


  /**
   * Set float key
   * @param key
   * @param value
   */
  public void set(String key, float value) {
    ConfigurationSet(key, value);
  }

  /**
   * Set double key
   * @param key
   * @param value
   */
  public void set(String key, double value) {
    ConfigurationSet(key, value);
  }

  /**
   * Set string key
   * @param key
   * @param value
   */
  public void set(String key, String value) {
    ConfigurationSet(key, value);
  }

  /**
   * Get string array by key
   * @param key
   * @return
   */
  public ArrayList<String> getStrings(String key, ArrayList<String> defaultValues) {
    return ConfigurationGetStrings(key, defaultValues);
  }

  /**
   * set string array
   * @param key
   * @param values
   */
  public void set(String key, ArrayList<String> values) {
    ConfigurationSet(key, values);
  }

  private native boolean ConfigurationGetBoolean(String key, boolean defaultValue);

  private native int ConfigurationGetInt(String key, int defaultValue);

  private native long ConfigurationGetLong(String key, long defaultValue);

  private native String ConfigurationGetString(String key, String defaultValue);

  private native float ConfigurationGetFloat(String key, float defaultValue);

  private native double ConfigurationGetDouble(String key, double defaultValue);

  private native void ConfigurationSet(String key, boolean value);

  private native void ConfigurationSet(String key, int value);

  private native void ConfigurationSet(String key, long value);

  private native void ConfigurationSet(String key, float value);

  private native void ConfigurationSet(String key, double value);

  private native void ConfigurationSet(String key, String value);

  private native ArrayList<String> ConfigurationGetStrings(String key,
      ArrayList<String> defaultValue);

  private native void ConfigurationSet(String key, ArrayList<String> values);

  private native void ConfigurationParser(String file);

  private native long ConfigurationNew();
}
