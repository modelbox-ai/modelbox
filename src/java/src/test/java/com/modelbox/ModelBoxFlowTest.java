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

import org.junit.Test;

public class ModelBoxFlowTest {
  @Test
  public void testFlowNotExist() throws Exception {
    String driverDir = TestConfig.TEST_DRIVER_DIR;

    String txt = "[driver]\n";
    txt += "dir=[\"" + driverDir + "\"]\n";
    txt += "skip-default=true\n";
    txt += "[log]\n";
    txt += "level=\"ERROR\"\n";
    txt += "[graph]\n";
    txt += "graphconf = '''digraph demo {{ \n";
    txt += "  notexist[type=flowunit, flowunit=notexist, device=cpu]\n";
    txt += "}}'''\n";
    txt += "format = \"graphviz\"\n";

    System.out.println(txt);
    Flow flow = new Flow();
    flow.init("NOT-EXIST", txt);
    flow.build();
  }
}