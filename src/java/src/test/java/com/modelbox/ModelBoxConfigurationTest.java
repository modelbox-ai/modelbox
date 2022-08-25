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
import java.util.ArrayList;
import org.junit.Test;

public class ModelBoxConfigurationTest {

  @Test
  public void testKeys() throws Exception {
    Configuration conf = new Configuration();
    conf.set("bool", true);
    conf.set("int", 1);
    conf.set("long", 1);
    conf.set("float", 1.1F);
    conf.set("double", 1.2);
    conf.set("string", "string");
    ArrayList<String> lists = new ArrayList<String>();
    lists.add("a");
    lists.add("b");
    lists.add("c");
    conf.set("strings", lists);

    assertEquals(conf.getBoolean("bool", false), true);
    assertEquals(conf.getInt("int", 0), 1);
    assertEquals(conf.getLong("long", 0), 1);
    assertEquals(conf.getFloat("float", 0.0F), 1.1F, 0.1);
    assertEquals(conf.getDouble("double", 0.0F), 1.2, 0.1);
    assertEquals(conf.getString("string", ""), "string");

    ArrayList<String> get_lists = conf.getStrings("strings", null);
    assertEquals(lists.size(), get_lists.size());

    for (int i = 0; i < lists.size(); i++) {
      assertEquals(lists.get(i), get_lists.get(i));
    }
  }

  @Test
  public void testKeysDefault() throws Exception {
    Configuration conf = new Configuration();

    ArrayList<String> lists = new ArrayList<String>();
    lists.add("a");
    lists.add("b");
    lists.add("c");
    conf.set("strings", lists);

    assertEquals(conf.getBoolean("bool", true), true);
    assertEquals(conf.getInt("int", 1), 1);
    assertEquals(conf.getLong("long", 1), 1);
    assertEquals(conf.getFloat("float", 1.1F), 1.1F, 0.1);
    assertEquals(conf.getDouble("double", 1.1F), 1.2, 0.1);
    assertEquals(conf.getString("string", "string"), "string");

    ArrayList<String> get_lists = conf.getStrings("strings", lists);
    assertEquals(lists, get_lists);
  }
}