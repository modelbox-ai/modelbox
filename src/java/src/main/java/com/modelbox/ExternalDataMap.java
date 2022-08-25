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

import java.util.HashMap;

/**
 * modelbox ExternalDataMap
 */
public class ExternalDataMap extends NativeObject {
  private Object priv_data = null;

  private ExternalDataMap() {
    // Create this object from Flow.
  }

  /**
   * Create buffer list 
   * @return bufferlist 
   */
  public BufferList CreateBufferList() {
    return ExternalDataMap_CreateBufferList();
  }

  /**
   * Set output data meta
   * @param name meta name
   * @param meta datameta object
   */
  public void setOutputMeta(String name, DataMeta meta) {
    ExternalDataMap_SetOutputMeta(name, meta);
  }

  /**
   * Send bufferlist to port
   * @param portName portname
   * @param bufferlist bufferlist 
   * @throws ModelBoxException
   */
  public void send(String portName, BufferList bufferlist) throws ModelBoxException {
    ExternalDataMap_Send(portName, bufferlist);
  }

  /**
   * Recv bufferlist map
   * @param timeout recv timeout in milliseconds,    
   *   timeout > 0 if no data blocking for timeout(ms) and return null.
   *   timeout = 0 if no data blocking until data is ready.
   *   timeout < 0 if no data return immediately. and return null.
   * @return output portname and bufferlist map
   * @throws ModelBoxException
   */
  public HashMap<String, BufferList> recv(long timeout) throws ModelBoxException {
    return ExternalDataMap_Recv(timeout);
  }

  /**
   * Recv bufferlist map, blocking until data is ready.
   * @return output portname and bufferlist map
   * @throws ModelBoxException
   */
  public HashMap<String, BufferList> recv() throws ModelBoxException {
    return recv(0);
  }

  /**
   * Close datamap, no input data anymore
   */
  public void close() {
    ExternalDataMap_Close();
  }

  /**
   * shutdown datamap, and exit.
   */
  public void shutdown() {
    ExternalDataMap_Shutdown();
  }

  /**
   * Set user private object
   * @param o
   */
  public void setPrivate(Object o) {
    priv_data = o;
  }

  /**
   * Get user private object
   * @param <T> user object type
   * @return user object
   */
  @SuppressWarnings("unchecked")
  public <T> T getPrivate() {
    try {
      return (T) priv_data;
    } catch (ClassCastException e) {
      return null;
    }
  }

  /**
   * Get session context
   * @return session context
   */
  public SessionContext getSessionContext() {
    return ExternalDataMap_GetSessionContext();
  }

  /**
   * Get sessioncontext configuration
   * @return sessioncontext configuration
   */
  public Configuration getSessionConfig() {
    return ExternalDataMap_GetSessionConfig();
  }

  /**
   * Get last error on datamap
   * @return flowunit error
   */
  public FlowUnitError getLastError() {
    return ExternalDataMap_GetLastError();
  }

  private native BufferList ExternalDataMap_CreateBufferList();

  private native void ExternalDataMap_SetOutputMeta(String name, DataMeta meta);

  private native void ExternalDataMap_Send(String portName, BufferList bufferList);

  private native HashMap<String, BufferList> ExternalDataMap_Recv(long timeout);

  private native void ExternalDataMap_Close();

  private native void ExternalDataMap_Shutdown();

  private native SessionContext ExternalDataMap_GetSessionContext();

  private native Configuration ExternalDataMap_GetSessionConfig();

  private native FlowUnitError ExternalDataMap_GetLastError();
}
