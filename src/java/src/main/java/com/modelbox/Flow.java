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
 * modelbox Flow
 */
public class Flow extends NativeObject {
  public Flow() {
    setNativeHandle(FlowNew());
  }

  /**
   * init flow from inline graph
   * @param name graph name
   * @param graph inline graph
   * @throws ModelBoxException
   */
  public void init(String name, String graph) throws ModelBoxException {
    FlowInit(name, graph);
  }

  /**
   * init flow from graph file
   * @param file path to graph file
   * @throws ModelBoxException
   */
  public void init(String file) throws ModelBoxException {
    FlowInit(file);
  }

  /**
   * init flow by name, args and flow directory
   * @param name flow name
   * @param args flow args
   * @param flowDir scan flow directory
   * @throws ModelBoxException
   */
  public void initByName(String name, Configuration args, String flowDir) throws ModelBoxException {
    FlowInitByName(name, args, flowDir);
  }

  /**
   * init flow by name, args and flow directory
   * @param name flow name
   * @param args flow args
   * @throws ModelBoxException
   */
  public void initByName(String name, Configuration args) throws ModelBoxException {
    FlowInitByName(name, args);
  }

  /**
   * init flow by name, args and flow directory
   * @param name flow name
   * @param args flow args
   * @throws ModelBoxException
   */
  public void initByName(String name) throws ModelBoxException {
    FlowInitByName(name, null);
  }

  /**
   * Start run flow
   * @throws ModelBoxException
   */
  public void startRun() throws ModelBoxException {
    FlowStartRun();
  }

  /**
   * Wait flow finish
   * @throws ModelBoxException
   */
  public void waitFor() throws ModelBoxException {
    waitFor(0);
  }

  /**
   * Wait flow finish
   * @param timeout wait timeout, in millisecond 
   * @return whether timeout
   * @throws ModelBoxException
   */
  public boolean waitFor(long timeout) throws ModelBoxException {
    Status retval = new Status();
    return waitFor(timeout, retval);
  }

  /**
   * Wait flor finish, and get flow result
   * @param timeout wait timeout, in millisecond 
   * @param retval flow result.
   * @return whether timeout
   * @throws ModelBoxException
   */
  public boolean waitFor(long timeout, Status retval) throws ModelBoxException {
    return FlowWait(timeout, retval);
  }

  /**
   * Stop flow
   * @throws ModelBoxException
   */
  public void stop() throws ModelBoxException {
    FlowStop();
  }

  /**
   * Create external data for sending data to flow
   * @return ExternalDataMap object
   * @throws ModelBoxException
   */
  public ExternalDataMap createExternalDataMap() throws ModelBoxException {
    return FlowCreateExternalDataMap();
  }

  /**
   * Create stream io to send and recv stream data
   * @return ExternalDataMap object
   * @throws ModelBoxException
   */
  public FlowStreamIO CreateStreamIO() throws ModelBoxException {
    return FlowCreateStreamIO();
  }

  private native long FlowNew();

  private native boolean FlowWait(long timeout, Status status);

  private native void FlowStartRun();

  private native void FlowInit(String name, String graph);

  private native void FlowInit(String file);

  private native void FlowInitByName(String name, Configuration args, String flowDir);

  private native void FlowInitByName(String name, Configuration args);

  private native void FlowStop();

  private native ExternalDataMap FlowCreateExternalDataMap();

  private native FlowStreamIO FlowCreateStreamIO();
}
