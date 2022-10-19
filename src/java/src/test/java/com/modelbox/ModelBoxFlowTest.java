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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import org.junit.BeforeClass;
import org.junit.Test;

public class ModelBoxFlowTest {

  public static class FlowUnitPassThrough extends FlowUnit {
    public static class Builder extends FlowUnitBuilder {
      @Override
      public void probe(FlowUnitDesc desc) throws ModelBoxException {
        desc.SetFlowUnitType("cpu");
        desc.SetFlowUnitName("javapassthrouth");
        desc.SetInputContiguous(false);
        desc.SetFlowType(FlowUnitDesc.FlowType.NORMAL);
        desc.AddFlowUnitInput(new FlowUnitInput("in"));
        desc.AddFlowUnitOutput(new FlowUnitOutput("out"));
      }

      @Override
      public FlowUnit build() throws ModelBoxException {
        return new FlowUnitPassThrough();
      }
    }

    public FlowUnitPassThrough() {}

    @Override
    public void open(Configuration opts) throws ModelBoxException {
      assertEquals(opts.getString("opt", ""), "value");
    }

    @Override
    public Status process(DataContext data_ctx) throws ModelBoxException {
      BufferList in = data_ctx.input("in");
      BufferList out = data_ctx.output("out");

      for (int i = 0; i < in.size(); i++) {
        out.pushBack(in.at(i));
      }

      return Status.OK();
    }
  }

  @BeforeClass
  public static void setUpTest() {
    Log.unRegLog();
    ModelBox.SetDefaultScanPath(TestConfig.TEST_DRIVER_DIR);
  }

  @Test(expected = ModelBoxException.Badconf.class)
  public void testFlowNotExist() throws Exception {
    String txt = "[log]\n";
    txt += "level=\"INFO\"\n";
    txt += "[graph]\n";
    txt += "graphconf = '''digraph demo {{ \n";
    txt += "  notexist[type=flowunit, flowunit=notexist, device=cpu]\n";
    txt += "}}'''\n";
    txt += "format = \"graphviz\"\n";

    System.out.println(txt);
    Flow flow = new Flow();
    flow.init("NOT-EXIST", txt);
    flow.startRun();
  }

  @Test
  public void testFlowProcessData() throws Exception {
    boolean get_result = false;
    String txt = "[log]\n";
    txt += "level=\"INFO\"\n";
    txt += "[graph]\n";
    txt += "graphconf = '''digraph demo {{ \n";
    txt += "  input[type=input] \n";
    txt += "  process[flowunit=passthrouth, device=cpu]\n";
    txt += "  output[type=output]\n";
    txt += "\n";
    txt += " input->process:in";
    txt += " process:out -> output\n";
    txt += "}}'''\n";
    txt += "format = \"graphviz\"\n";

    System.out.println(txt);
    Flow flow = new Flow();
    flow.init("Process", txt);
    flow.startRun();
    ExternalDataMap datamap = flow.createExternalDataMap();
    BufferList data = datamap.CreateBufferList();
    assertEquals(data.getDevice().getType(), "cpu");

    data.build(new int[] {0});
    String msg = "Hello world";
    data.at(0).build(msg.getBytes());
    data.pushBack(msg.getBytes());
    datamap.send("input", data);
    datamap.close();
    datamap.setPrivate("this is a test");
    Log.info("session id is " + datamap.getSessionContext().getSessionId());

    ExternalDataSelect data_select = new ExternalDataSelect();
    data_select.register(datamap);

    while (true) {
      try {
        ArrayList<ExternalDataMap> datamaplist = data_select.select(1000 * 10);
        if (datamaplist == null) {
          assertFalse(true);
          break;
        }

        for (ExternalDataMap outdatamap : datamaplist) {
          System.out.println("Get: " + outdatamap.getPrivate());
          HashMap<String, BufferList> outdata = outdatamap.recv();
          if (outdata == null) {
            data_select.remove(outdatamap);
            throw new ModelBoxException.Eof("exit");
          }

          assertEquals(outdata.size(), 1);
          assertEquals(datamap, outdatamap);
          for (Map.Entry<String, BufferList> entry : outdata.entrySet()) {
            String key = entry.getKey();
            BufferList value = entry.getValue();
            assertEquals(key, "output");
            assertEquals(value.size(), 2);
            String str = new String(value.at(0).getData());
            assertEquals(msg, str);
            Log.info("Message is: " + str);
            get_result = true;
          }
        }
      } catch (ModelBoxException.Eof e) {
        break;
      } catch (ModelBoxException e) {
        System.out.println("select failed, " + e.getMessage());
        assertFalse(true);
        break;
      }
    }

    assertTrue(get_result);
    flow = null;
    System.gc();
  }

  @Test
  public void testFlowStreamIO() throws Exception {
    boolean get_result = false;
    String txt = "[log]\n";
    txt += "level=\"INFO\"\n";
    txt += "[graph]\n";
    txt += "graphconf = '''digraph demo {{ \n";
    txt += "  input[type=input] \n";
    txt += "  process[flowunit=passthrouth, device=cpu]\n";
    txt += "  output[type=output]\n";
    txt += "\n";
    txt += " input->process:in";
    txt += " process:out -> output\n";
    txt += "}}'''\n";
    txt += "format = \"graphviz\"\n";
    

    System.out.println(txt);
    Flow flow = new Flow();
    flow.init("Process", txt);
    flow.startRun();
    FlowStreamIO streamio = flow.CreateStreamIO();
    Buffer data = streamio.createBuffer();
    assertEquals(data.getDevice().getType(), "cpu");
    String msg = "Hello world";
    data.build(msg.getBytes());
    streamio.send("input", data);
    streamio.send("input", msg.getBytes());
    streamio.closeInput();
    int count = 0;
    
    while (true) {
      Buffer outdata = streamio.recv("output", 1000 * 10);
      if (outdata == null) {
        break;
      }
      
      String str = new String(outdata.getData());
      assertEquals(msg, str);
      Log.info("Message is: " + str);
      get_result = true;
      count++;
    }
    
    assertEquals(count, 2);
    assertTrue(get_result);
    flow = null;
    System.gc();
  }

  @Test
  public void testFlowRegister() throws Exception {
    boolean get_result = false;
    String txt = "[log]\n";
    txt += "level=\"INFO\"\n";
    txt += "[graph]\n";
    txt += "graphconf = '''digraph demo {{ \n";
    txt += "  input[type=input] \n";
    txt += "  process[flowunit=javapassthrouth, device=cpu, opt=value]\n";
    txt += "  output[type=output]\n";
    txt += "\n";
    txt += " input->process:in";
    txt += " process:out -> output\n";
    txt += "}}'''\n";
    txt += "format = \"graphviz\"\n";
    

    System.out.println(txt);
    Flow flow = new Flow();
    flow.RegisterFlowUnit(new FlowUnitPassThrough.Builder());
    flow.init("Process", txt);
    flow.startRun();
    FlowStreamIO streamio = flow.CreateStreamIO();
    Buffer data = streamio.createBuffer();
    assertEquals(data.getDevice().getType(), "cpu");
    String msg = "Hello world";
    data.build(msg.getBytes());
    streamio.send("input", data);
    streamio.send("input", msg.getBytes());
    streamio.closeInput();
    int count = 0;
    
    while (true) {
      Buffer outdata = streamio.recv("output", 1000 * 10);
      if (outdata == null) {
        break;
      }
      
      String str = new String(outdata.getData());
      assertEquals(msg, str);
      Log.info("Message is: " + str);
      get_result = true;
      count++;
    }
    
    assertEquals(count, 2);
    assertTrue(get_result);
    flow = null;
    System.gc();
  }
}
