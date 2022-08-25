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

public class DataContext extends NativeObject {
  private DataContext() {

  }

  /**
   * Get input port bufferlist
   * @param portName port name
   * @return BufferList
   */
  public BufferList input(String portName) {
    return DataContext_Input(portName);
  }

  /**
   * Get output port bufferlist
   * @param portName port name
   * @return BufferList
   */
  public BufferList output(String portName) {
    return DataContext_Output(portName);
  }

  /**
   * Get external port bufferlist
   * @return BufferList
   */
  public BufferList external() {
    return DataContext_External();
  }

  /**
   * Has error
   * @return boolean
   */
  public boolean hasError() {
    return DataContext_HasError();
  }

  /**
   * Send event to flowunit
   * @param event
   */
  public void sendEvent(FlowUnitEvent event) {
    DataContext_SendEvent(event);
  }

  /**
   * Set private 
   * @param key private key
   * @param priv private object
   */
  public void setPrivate(String key, Object priv) {
    DataContext_SetPrivate(key, priv);
  }

  /**
   * Get private
   * @param key private key
   * @return object
   */
  public Object getPrivate(String key) {
    return DataContext_GetPrivate(key);
  }

  /**
   * Get input meta
   * @param portName portname
   * @return data meta
   */
  public DataMeta getInputMeta(String portName) {
    return DataContext_GetInputMeta(portName);
  }

  /**
   * Set output meta
   * @param portName portname
   * @param dataMeta data meta
   */
  public void setOutputMeta(String portName, DataMeta dataMeta) {
    DataContext_SetOututMeta(portName, dataMeta);
  }

  /**
   * Get session context
   * @return session context
   */
  public SessionContext getSessionContext() {
    return DataContext_GetSessionContext();
  }

  /**
   * Get session configuration
   * @return session configuration
   */
  public Configuration getSessionConfig() {
    return DataContext_GetSessionConfig();
  }

  /**
   * get Statistics
   * @return Statistics
   */
  public StatisticsItem getStatistics() {
    return DataContext_GetStatistics();
  }

  private native BufferList DataContext_Input(String portName);

  private native BufferList DataContext_Output(String portName);

  private native BufferList DataContext_External();

  private native boolean DataContext_HasError();

  private native void DataContext_SendEvent(FlowUnitEvent event);

  private native void DataContext_SetPrivate(String key, Object priv);

  private native Object DataContext_GetPrivate(String key);

  private native DataMeta DataContext_GetInputMeta(String portName);

  private native void DataContext_SetOututMeta(String portName, DataMeta dataMeta);

  private native SessionContext DataContext_GetSessionContext();

  private native Configuration DataContext_GetSessionConfig();

  private native StatisticsItem DataContext_GetStatistics();
}
