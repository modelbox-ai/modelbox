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
 * modelbox FlowStreamIO
 */
public class FlowStreamIO extends NativeObject {
  private FlowStreamIO() {

  }

  /**
   * create a empty buffer on cpu device
   * @returncpu buffer
   * @throws ModelBoxException
   */
  public Buffer createBuffer() throws ModelBoxException {
    return FlowStreamIO_CreateBuffer();
  }

  /**
   * Send buffer of this stream to flow
   * @param inputName input node name of flow
   * @param buffer buffer of this stream
   * @throws ModelBoxException
   */
  public void send(String inputName, Buffer buffer) throws ModelBoxException {
    FlowStreamIO_Send(inputName, buffer);
  }

  /**
   * Send buffer of this stream to flow
   * @param inputName input node name of flow
   * @param data data of this stream
   * @throws ModelBoxException
   */
  public void send(String inputName, byte[] data) throws ModelBoxException {
    FlowStreamIO_Send(inputName, data);
  }

  /**
   * @brief recv buffer of this stream result from flow
   * @param output_name output node name of flow
   * @param buffer result buffer of this stream
   * @param timeout wait result timeout
   * @return Status
   **/
  /**
   * Recv buffer of this stream result from flow
   * @param outputName output node name of flow
   * @param timeout wait result timeout
   *   timeout > 0 if no data blocking for timeout(ms) and return null.
   *   timeout = 0 if no data blocking until data is ready.
   *   timeout < 0 if no data return immediately. and return null.
   * @return result buffer of this stream
   * @throws ModelBoxException
   */
  public Buffer recv(String outputName, long timeout) throws ModelBoxException {
    return FlowStreamIO_Recv(outputName, timeout);
  }

  /**
   * Close input stream, mark stream end
   */
  public void closeInput() {
    FlowStreamIO_CloseInput();
  }

  private native Buffer FlowStreamIO_CreateBuffer();

  private native void FlowStreamIO_Send(String inputName, Buffer buffer);

  private native void FlowStreamIO_Send(String inputName, byte[] data);

  private native Buffer FlowStreamIO_Recv(String outputName, long timeout);

  private native void FlowStreamIO_CloseInput();
}
