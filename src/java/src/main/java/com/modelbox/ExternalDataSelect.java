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
 * modelbox ExternalDataSelect
 */
public class ExternalDataSelect extends NativeObject {
  public ExternalDataSelect() {
    setNativeHandle(ExternalDataSelect_New());
  }

  /**
   * Register ExternalDataMap to ExternalDataSelect 
   * @param dataMap datamap object
   */
  public void register(ExternalDataMap dataMap) {
    ExternalDataSelect_RegisterExternalData(dataMap);
  }

  /**
   * From ExternalDataMap from ExternalDataSelect 
   * @param dataMap datamap object
   */
  public void remove(ExternalDataMap dataMap) {
    ExternalDataSelect_RemoveExternalData(dataMap);
  }

  /**
   * Wait from datamap ready
   * @param timeout wait timeout. if timeout < 0, wait until data ready.
   * @return datamap ready to recv
   * @throws ModelBoxException
   */
  public ArrayList<ExternalDataMap> select(long timeout) throws ModelBoxException {
    return ExternalDataSelect_SelectExternalData(timeout);
  }

  /**
   * Wait from datamap ready
   * @param timeout wait timeout. wait until data ready.
   * @return datamap ready to recv
   * @throws ModelBoxException
   */
  public ArrayList<ExternalDataMap> select() throws ModelBoxException {
    return select(-1);
  }

  private native long ExternalDataSelect_New();

  private native void ExternalDataSelect_RegisterExternalData(ExternalDataMap dataMap);

  private native void ExternalDataSelect_RemoveExternalData(ExternalDataMap dataMap);

  private native ArrayList<ExternalDataMap> ExternalDataSelect_SelectExternalData(long timeout);
}
