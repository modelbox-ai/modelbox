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
import org.junit.BeforeClass;
import org.junit.Test;

public class ModelboxBufferTest {
  @BeforeClass
  public static void setUpTest() {
    Log.unRegLog();
    ModelBox.SetDefaultScanPath(TestConfig.TEST_DRIVER_DIR);
  }

  @Test
  public void testBufferMeta() throws Exception {
    String txt = "[log]\n";
    txt += "level=\"INFO\"\n";
    txt += "[graph]\n";
    txt += "graphconf = '''digraph demo {{ \n";
    txt += " input[type=input]\n";
    txt += " output[type=output]\n";
    txt += "input -> output\n";
    txt += "}}'''\n";
    txt += "format = \"graphviz\"\n";

    System.out.println(txt);
    Flow flow = new Flow();
    flow.init("NOT-EXIST", txt);
    flow.startRun();
    FlowStreamIO streamio = flow.CreateStreamIO();
    Buffer data = streamio.createBuffer();
    data.setMetaInt("int", 1);
    data.setMetaFloat("float", 1.0f);
    data.setMetaString("string", "1");
    data.setMetaLong("long", 2);
    data.setMetaDouble("double", 2.0);
    assertEquals(data.getMetaInt("int"), 1);
    assertEquals(data.getMetaFloat("float"), 1.0f, 0.1);
    assertEquals(data.getMetaString("string"), "1");
    assertEquals(data.getMetaLong("long"), 2);
    assertEquals(data.getMetaDouble("double"), 2.0, 0.1);
  }
}
