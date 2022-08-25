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

import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class ModelBoxMiscTest {
  @Test
  public void testFlowUnitError() throws Exception {
    FlowUnitError err = new FlowUnitError("desc");
    assertEquals(err.getDesc(), "desc");

    Status status = new Status(StatusCode.STATUS_ALREADY, "status");
    err = new FlowUnitError("a", "b", status);
    assertEquals(err.getDesc(), "node:a error pos:b status:Operation already in progress error:status");
  }

  @Test
  public void testFlowUnitDataMeta() throws Exception {
    DataMeta data = new DataMeta();
    data.set("a", "a");
    data.set("b", "b");

    assertEquals(data.getString("a"), "a");
    assertEquals(data.getString("b"), "b");
    assertEquals(data.getString("c"), null);
  }

  @Test
  public void testFlowUnitEvent() throws Exception {
    FlowUnitEvent event = new FlowUnitEvent();
    event.set("a", "a");
    event.set("b", "b");


    assertEquals(event.get("a"), "a");
    assertEquals(event.get("b"), "b");
    assertEquals(event.get("c"), null);
  }
}