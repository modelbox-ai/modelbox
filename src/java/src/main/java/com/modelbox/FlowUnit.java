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

abstract public class FlowUnit extends NativeObject {
  public FlowUnit() {
    setNativeHandle(FlowUnit_New());
  }

  /**
   * Flowunit Open 
   * @param opts
   * @return 
   */
  public void open(Configuration opts) throws ModelBoxException {}

  /**
   * Flowunit Close
   * @return
   */
  public void close() throws ModelBoxException {}

  /**
   * FlowUnit data process
   * @param data_ctx
   * @return
   */
  abstract public Status process(DataContext data_ctx) throws ModelBoxException;

  /**
   * Flowunit data pre
   * @param data_ctx
   * @return
   */
  public void dataPre(DataContext data_ctx) throws ModelBoxException {}

  /**
   * FlowUnit data Post;
   * @param data_ctx
   * @return
   */
  public void dataPost(DataContext data_ctx) throws ModelBoxException {}

  private native long FlowUnit_New();
}
