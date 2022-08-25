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

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.json.JSONObject;

public class TestConfig {

  static {
    /**
     for vscode junit:
     add the following settings in settings.json
          
     "java.test.config": {
        "vmArgs": [
            "-Djava.library.path=${workspaceFolder}/build/src/java/jni"
        ],
        "env" : {
            "TEST_CONFIG_JSON_FILE" : "${workspaceFolder}/build/src/java/src/test/java/com/modelbox/TestConfig.json"
        }
      },
     */
    String jsonfile = System.getenv("TEST_CONFIG_JSON_FILE");
    if (jsonfile != null) {
      try {
        JSONObject jsonobject = new JSONObject(
            new String(Files.readAllBytes(Paths.get(jsonfile)), StandardCharsets.UTF_8));
        TestConfig.TEST_WORKING_DIR = jsonobject.getString("TEST_WORKING_DIR");
        TestConfig.TEST_LIB_DIR = jsonobject.getString("TEST_LIB_DIR");
        TestConfig.TEST_BIN_DIR = jsonobject.getString("TEST_BIN_DIR");
        TestConfig.TEST_DATA_DIR = jsonobject.getString("TEST_DATA_DIR");
        TestConfig.TEST_SOURCE_DIR = jsonobject.getString("TEST_SOURCE_DIR");
        TestConfig.TEST_DRIVER_DIR = jsonobject.getString("TEST_DRIVER_DIR");
        TestConfig.TEST_ASSETS = jsonobject.getString("TEST_ASSETS");
      } catch (Exception e) {
        System.err.println("Load json file " + jsonfile + " failed");
      }
    }
  }

  static String TEST_WORKING_DIR;

  static String TEST_LIB_DIR;

  static String TEST_BIN_DIR;

  static String TEST_DATA_DIR;

  static String TEST_SOURCE_DIR;

  static String TEST_DRIVER_DIR;

  static String TEST_ASSETS;
}
