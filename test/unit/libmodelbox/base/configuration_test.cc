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

#include "modelbox/base/configuration.h"

#include <fstream>

#include "gtest/gtest.h"
#include "test_config.h"

namespace modelbox {
class ConfigurationTest : public testing::Test {
 public:
  ConfigurationTest() = default;

 protected:
  void SetUp() override{

  };
  void TearDown() override{};
};

TEST_F(ConfigurationTest, AddPropertyTest) {
  ConfigurationBuilder builder;
  std::map<std::string, std::string> values = {{"1", "aaa"}, {"2", "bbb"}};

  builder.AddProperty("graph.name", "111111");
  builder.AddProperty("graph.name", "123123");
  builder.AddProperty("graph.node.index", "abcabc");
  builder.AddProperties(values);
  auto config = builder.Build();

  EXPECT_TRUE(config->Contain("graph.name"));
  EXPECT_FALSE(config->Contain("graph.nameX"));
  EXPECT_EQ(config->GetString("graph.name"), "123123");
  EXPECT_EQ(config->GetString("graph.node.index"), "abcabc");
  EXPECT_EQ(config->GetString("graph.nokey"), "");
  EXPECT_EQ(config->GetString("1"), "aaa");
  EXPECT_EQ(config->GetString("2"), "bbb");
  EXPECT_EQ(config->GetString("3", "cc"), "cc");
}

TEST_F(ConfigurationTest, GetBoolTest) {
  ConfigurationBuilder builder;
  builder.AddProperty("1", "false");
  builder.AddProperty("2", "true");
  builder.AddProperty("3", "0");
  builder.AddProperty("4", "1");
  builder.AddProperty("5", "!D!DSA");
  builder.AddProperty("6", "01s12");
  builder.AddProperty("7", "");
  auto config = builder.Build();

  EXPECT_FALSE(config->GetBool("1"));
  EXPECT_TRUE(config->GetBool("2"));
  EXPECT_FALSE(config->GetBool("3"));
  EXPECT_TRUE(config->GetBool("4"));
  EXPECT_FALSE(config->GetBool("5"));
  EXPECT_TRUE(config->GetBool("6", true));
  EXPECT_FALSE(config->GetBool("7"));
  EXPECT_FALSE(config->GetBool("8"));
  EXPECT_TRUE(config->GetBool("9", true));
}

TEST_F(ConfigurationTest, GetIntTest) {
  ConfigurationBuilder builder;
  builder.AddProperty("1", "12aa12");
  builder.AddProperty("2", "123a");
  builder.AddProperty("3", "123.0");
  builder.AddProperty("4", "123-0");
  builder.AddProperty("5", "0x123");
  builder.AddProperty("6", "123b");
  builder.AddProperty("7", "");
  builder.AddProperty("8", "a!@");
  builder.AddProperty("9", "123.123.123");
  builder.AddProperty("10", "99999999999999999999");

  auto invalidConfig = builder.Build();

  for (size_t i = 0; i <= invalidConfig->Size(); ++i) {
    EXPECT_EQ(invalidConfig->GetInt8(std::to_string(i)), 0);
    EXPECT_EQ(invalidConfig->GetUint8(std::to_string(i)), 0);
    EXPECT_EQ(invalidConfig->GetInt16(std::to_string(i)), 0);
    EXPECT_EQ(invalidConfig->GetUint16(std::to_string(i)), 0);
    EXPECT_EQ(invalidConfig->GetInt32(std::to_string(i)), 0);
    EXPECT_EQ(invalidConfig->GetUint32(std::to_string(i)), 0);
    EXPECT_EQ(invalidConfig->GetInt64(std::to_string(i)), 0);
    EXPECT_EQ(invalidConfig->GetUint64(std::to_string(i)), 0);
  }

  std::map<std::string, std::string> range_test = {
      {"0", "0"},
      {"1", "+1"},
      {"2", "-1"},
      {"3", "127"},
      {"4", "-128"},
      {"5", "255"},
      {"6", "32767"},
      {"7", "-32768"},
      {"8", "65535"},
      {"9", "2147483647"},
      {"10", "-2147483648"},
      {"11", "4294967295"},
      {"12", "9223372036854775807"},
      {"13", "-9223372036854775808"},
      {"14", "18446744073709551615"}};
  builder.AddProperties(range_test);
  auto range_test_config = builder.Build();

  std::vector<int8_t> int8_result = {0, 1, -1, INT8_MAX, INT8_MIN, 0, 0, 0,
                                     0, 0, 0,  0,        0,        0, 0};
  std::vector<uint8_t> uint8_result = {0, 1, 0, INT8_MAX, 0, UINT8_MAX, 0, 0,
                                       0, 0, 0, 0,        0, 0,         0};
  std::vector<int16_t> int16_result = {
      0, 1, -1, INT8_MAX, INT8_MIN, UINT8_MAX, INT16_MAX, INT16_MIN,
      0, 0, 0,  0,        0,        0,         0};
  std::vector<uint16_t> uint16_result = {
      0,          1, 0, INT8_MAX, 0, UINT8_MAX, INT16_MAX, 0,
      UINT16_MAX, 0, 0, 0,        0, 0,         0};
  std::vector<int32_t> int32_result = {
      0,         1,         -1,        INT8_MAX,   INT8_MIN,
      UINT8_MAX, INT16_MAX, INT16_MIN, UINT16_MAX, INT32_MAX,
      INT32_MIN, 0,         0,         0,          0};
  std::vector<uint32_t> uint32_result = {
      0,          1,         0, INT8_MAX,   0, UINT8_MAX, INT16_MAX, 0,
      UINT16_MAX, INT32_MAX, 0, UINT32_MAX, 0, 0,         0};
  std::vector<int64_t> int64_result = {
      0,         1,          -1,        INT8_MAX,   INT8_MIN,
      UINT8_MAX, INT16_MAX,  INT16_MIN, UINT16_MAX, INT32_MAX,
      INT32_MIN, UINT32_MAX, INT64_MAX, INT64_MIN,  0};
  std::vector<uint64_t> uint64_result = {
      0,          1,         0, INT8_MAX,   0,         UINT8_MAX, INT16_MAX, 0,
      UINT16_MAX, INT32_MAX, 0, UINT32_MAX, INT64_MAX, 0,         UINT64_MAX};

  for (size_t i = 0; i < range_test_config->Size(); ++i) {
    EXPECT_EQ(range_test_config->GetInt8(std::to_string(i)), int8_result[i]);
    EXPECT_EQ(range_test_config->GetUint8(std::to_string(i)), uint8_result[i]);
    EXPECT_EQ(range_test_config->GetInt16(std::to_string(i)), int16_result[i]);
    EXPECT_EQ(range_test_config->GetUint16(std::to_string(i)),
              uint16_result[i]);
    EXPECT_EQ(range_test_config->GetInt32(std::to_string(i)), int32_result[i]);
    EXPECT_EQ(range_test_config->GetUint32(std::to_string(i)),
              uint32_result[i]);
    EXPECT_EQ(range_test_config->GetInt64(std::to_string(i)), int64_result[i]);
    EXPECT_EQ(range_test_config->GetUint64(std::to_string(i)),
              uint64_result[i]);
  }
}

TEST_F(ConfigurationTest, GetFloatTest) {
  ConfigurationBuilder builder;
  builder.AddProperty("1", "12aa12");
  builder.AddProperty("2", "123a");
  builder.AddProperty("3", "123.0+2");
  builder.AddProperty("4", "1.23-0");
  builder.AddProperty("5", "0.x123");
  builder.AddProperty("6", "123.b");
  builder.AddProperty("7", "");
  builder.AddProperty("8", "a!@");
  builder.AddProperty("9", "123.123.123");
  auto invalidConfig = builder.Build();

  for (size_t i = 0; i <= invalidConfig->Size(); ++i) {
    EXPECT_EQ(invalidConfig->GetFloat(std::to_string(i)), 0);
    EXPECT_EQ(invalidConfig->GetDouble(std::to_string(i)), 0);
  }

  builder.AddProperty("0", "0");
  builder.AddProperty("1", "1");
  builder.AddProperty("2", "-1");
  builder.AddProperty("3", "123456789");
  builder.AddProperty("4", "-123456789");
  builder.AddProperty("5", "12345.6789");
  builder.AddProperty("6", "-12345.6789");
  builder.AddProperty("7", "1.7e+30");
  builder.AddProperty("8", "-1.7e+30");
  builder.AddProperty("9", "1.7e-30");
  builder.AddProperty("10", "-1.7e-30");
  auto config = builder.Build();

  std::vector<float> float_result = {
      0,           1,       -1,       123456789.0, -123456789.0, 12345.6789,
      -12345.6789, 1.7e+30, -1.7e+30, 1.7e-30,     -1.7e-30};
  std::vector<double> double_result = {
      0,           1,       -1,       123456789.0, -123456789.0, 12345.6789,
      -12345.6789, 1.7e+30, -1.7e+30, 1.7e-30,     -1.7e-30};

  for (size_t i = 0; i < config->Size(); ++i) {
    EXPECT_EQ(config->GetFloat(std::to_string(i)), float_result[i]);
    EXPECT_EQ(config->GetDouble(std::to_string(i)), double_result[i]);
  }
}

TEST_F(ConfigurationTest, GetVectorTest) {
  ConfigurationBuilder builder;
  builder.AddProperty("1", std::string("1") + LIST_DELIMITER + "0" +
                               LIST_DELIMITER + "true" + LIST_DELIMITER +
                               "false");
  builder.AddProperty("2", std::string("1") + LIST_DELIMITER + "0" +
                               LIST_DELIMITER + "true" + LIST_DELIMITER +
                               "false" + LIST_DELIMITER + "g");
  builder.AddProperty("3", std::string("1") + LIST_DELIMITER + "0" +
                               LIST_DELIMITER + "-3" + LIST_DELIMITER + "5" +
                               LIST_DELIMITER + "-9");
  std::vector<int8_t> int_result = {1, 0, -3, 5, -9};
  builder.AddProperty("4", std::string("1") + LIST_DELIMITER + "0" +
                               LIST_DELIMITER + "3" + LIST_DELIMITER + "5" +
                               LIST_DELIMITER + "22");
  std::vector<uint8_t> uint_result = {1, 0, 3, 5, 22};
  builder.AddProperty("5", std::string("1.0") + LIST_DELIMITER + "0.0" +
                               LIST_DELIMITER + "3.45645641" + LIST_DELIMITER +
                               "551631.13124" + LIST_DELIMITER + "-22e+10");
  std::vector<float> float_result = {1.0, 0, 3.45645641, 551631.13124, -22e+10};
  std::vector<double> double_result = {1.0, 0, 3.45645641, 551631.13124,
                                       -22e+10};

  auto config = builder.Build();

  auto strings = config->GetStrings("1");
  EXPECT_EQ(strings.size(), 4);
  EXPECT_EQ(strings[0], "1");
  EXPECT_EQ(strings[1], "0");
  EXPECT_EQ(strings[2], "true");
  EXPECT_EQ(strings[3], "false");

  auto bools = config->GetBools("1");
  EXPECT_EQ(bools.size(), 4);
  EXPECT_TRUE(bools[0]);
  EXPECT_FALSE(bools[1]);
  EXPECT_TRUE(bools[2]);
  EXPECT_FALSE(bools[3]);
  EXPECT_EQ(config->GetBools("2").size(), 0);

  auto int8s = config->GetInt8s("3");
  auto uint8s = config->GetUint8s("4");
  auto int16s = config->GetInt16s("3");
  auto uint16s = config->GetUint16s("4");
  auto int32s = config->GetInt32s("3");
  auto uint32s = config->GetUint32s("4");
  auto int64s = config->GetInt64s("3");
  auto uint64s = config->GetUint64s("4");
  auto floats = config->GetFloats("5");
  auto doubles = config->GetDoubles("5");
  EXPECT_EQ(int8s.size(), 5);
  EXPECT_EQ(uint8s.size(), 5);
  EXPECT_EQ(int16s.size(), 5);
  EXPECT_EQ(uint16s.size(), 5);
  EXPECT_EQ(int32s.size(), 5);
  EXPECT_EQ(uint32s.size(), 5);
  EXPECT_EQ(int64s.size(), 5);
  EXPECT_EQ(uint64s.size(), 5);
  EXPECT_EQ(floats.size(), 5);
  EXPECT_EQ(doubles.size(), 5);
  for (size_t i = 0; i < int8s.size(); ++i) {
    EXPECT_EQ(int8s[i], int_result[i]);
    EXPECT_EQ(uint8s[i], uint_result[i]);
    EXPECT_EQ(int16s[i], int_result[i]);
    EXPECT_EQ(uint16s[i], uint_result[i]);
    EXPECT_EQ(int32s[i], int_result[i]);
    EXPECT_EQ(uint32s[i], uint_result[i]);
    EXPECT_EQ(int64s[i], int_result[i]);
    EXPECT_EQ(uint64s[i], uint_result[i]);
    EXPECT_EQ(floats[i], float_result[i]);
    EXPECT_EQ(doubles[i], double_result[i]);
  }

  EXPECT_EQ(config->GetInt8s("1").size(), 0);
  EXPECT_EQ(config->GetUint8s("1").size(), 0);
  EXPECT_EQ(config->GetInt16s("1").size(), 0);
  EXPECT_EQ(config->GetUint16s("1").size(), 0);
  EXPECT_EQ(config->GetInt32s("1").size(), 0);
  EXPECT_EQ(config->GetUint32s("1").size(), 0);
  EXPECT_EQ(config->GetInt64s("1").size(), 0);
  EXPECT_EQ(config->GetUint64s("1").size(), 0);
  EXPECT_EQ(config->GetFloats("1").size(), 0);
  EXPECT_EQ(config->GetDoubles("1").size(), 0);
}

TEST_F(ConfigurationTest, SetPropertyTest) {
  ConfigurationBuilder builder;
  auto config = builder.Build();
  config->SetProperty("1", 1);
  config->SetProperty("2", 1.2F);
  config->SetProperty("3", 1.3);
  config->SetProperty("4", false);
  config->SetProperty("5", true);
  config->SetProperty("6", "test");

  EXPECT_EQ(config->GetString("1"), "1");
  EXPECT_FLOAT_EQ(config->GetFloat("2"), 1.2F);
  EXPECT_FLOAT_EQ(config->GetDouble("3"), 1.3);
  EXPECT_EQ(config->GetString("4"), "0");
  EXPECT_EQ(config->GetString("5"), "1");
  EXPECT_EQ(config->GetString("6"), "test");

  config->SetProperty("5", std::vector<int32_t>{1, 2, 3});
  auto float_list = std::vector<float>{1.1F, 2.2F, 3.3F};
  config->SetProperty("6", float_list);

  EXPECT_EQ(config->GetString("5"),
            std::string("1") + LIST_DELIMITER + "2" + LIST_DELIMITER + "3");
  auto res = config->GetFloats("6");
  EXPECT_EQ(res.size(), float_list.size());
  for (size_t i = 0; i < float_list.size(); ++i) {
    EXPECT_FLOAT_EQ(res[i], float_list[i]);
  }
}

TEST_F(ConfigurationTest, SetPropertyWithoutBuilderTest) {
  Configuration config;
  config.SetProperty("1", 1);
  config.SetProperty("2", 1.2F);
  config.SetProperty("3", 1.3);
  config.SetProperty("4", false);
  config.SetProperty("5", true);
  config.SetProperty("6", "test");

  EXPECT_EQ(config.GetString("1"), "1");
  EXPECT_FLOAT_EQ(config.GetFloat("2"), 1.2F);
  EXPECT_FLOAT_EQ(config.GetDouble("3"), 1.3);
  EXPECT_EQ(config.GetString("4"), "0");
  EXPECT_EQ(config.GetString("5"), "1");
  EXPECT_EQ(config.GetString("6"), "test");
}

TEST_F(ConfigurationTest, GetSubKeysTest) {
  ConfigurationBuilder builder;
  auto config = builder.Build();
  EXPECT_EQ(StatusError, STATUS_SUCCESS);

  config->SetProperty("graph.node.1", 1);
  config->SetProperty("graph.node.2", 1.2F);
  config->SetProperty("device.gpu.0", 1.3);
  config->SetProperty("device.gpu.1", false);
  config->SetProperty("graph.edge.in.1", true);
  config->SetProperty("graph.edge.in.2", "test");
  config->SetProperty("graph.edge.out.1", "test");
  config->SetProperty("graph.edge.out.2", "test");
  config->SetProperty("graph.edge.out.3", "test");

  auto res = config->GetSubKeys("graph");
  EXPECT_EQ(res.size(), 2);
  std::set<std::string> expect_res = {"node", "edge"};
  auto is_equal = std::equal(res.begin(), res.end(), expect_res.begin());
  EXPECT_TRUE(is_equal);

  res = config->GetSubKeys("graph.edge");
  EXPECT_EQ(res.size(), 2);
  expect_res = {"in", "out"};
  is_equal = std::equal(res.begin(), res.end(), expect_res.begin());
  EXPECT_TRUE(is_equal);

  res = config->GetSubKeys("graph.edge.out");
  EXPECT_EQ(res.size(), 3);
  expect_res = {"1", "2", "3"};
  is_equal = std::equal(res.begin(), res.end(), expect_res.begin());
  EXPECT_TRUE(is_equal);

  res = config->GetSubKeys("graph.");
  EXPECT_EQ(res.size(), 0);

  res = config->GetSubKeys("graph.nod");
  EXPECT_EQ(res.size(), 0);

  res = config->GetSubKeys("graph.node.1");
  EXPECT_EQ(res.size(), 0);

  res = config->GetSubKeys("graph.node.1.2");
  EXPECT_EQ(res.size(), 0);

  res = config->GetSubKeys("");
  EXPECT_EQ(res.size(), 0);

  res = config->GetSubKeys("node");
  EXPECT_EQ(res.size(), 0);
}

TEST_F(ConfigurationTest, GetConfigKeysTest) {
  ConfigurationBuilder builder;
  builder.AddProperties({{"1", "1"}, {"2", "2"}, {"3", "3"}});
  auto config = builder.Build();
  EXPECT_EQ(StatusError, STATUS_SUCCESS);

  auto keys = config->GetKeys();
  EXPECT_EQ(keys.size(), 3);
  EXPECT_NE(keys.find("1"), keys.end());
  EXPECT_NE(keys.find("2"), keys.end());
  EXPECT_NE(keys.find("3"), keys.end());
  EXPECT_EQ(keys.find("4"), keys.end());
}

TEST_F(ConfigurationTest, GetSubConfigTest) {
  ConfigurationBuilder builder;
  auto config = builder.Build();
  EXPECT_EQ(StatusError, STATUS_SUCCESS);

  config->SetProperty("graph.node.1", 1);
  config->SetProperty("graph.node.2", 1.2F);
  config->SetProperty("device.gpu.0", 1.3);
  config->SetProperty("device.gpu.1", false);
  config->SetProperty("graph.edge.in.1", true);
  config->SetProperty("graph.edge.in.2", "test");
  config->SetProperty("graph.edge.out.1", "test");
  config->SetProperty("graph.edge.out.2", "test");
  config->SetProperty("graph.edge.out.3", "test");

  auto sub_config = config->GetSubConfig("device");
  EXPECT_EQ(sub_config->Size(), 2);
  EXPECT_EQ(sub_config->GetString("gpu.0"), "1.3");
  EXPECT_EQ(sub_config->GetString("gpu.1"), "0");
  EXPECT_EQ(sub_config->GetString("node.1"), "");

  sub_config = config->GetSubConfig("graph.edge.out");
  EXPECT_EQ(sub_config->Size(), 3);
  EXPECT_EQ(sub_config->GetString("1"), "test");
  EXPECT_EQ(sub_config->GetString("2"), "test");
  EXPECT_EQ(sub_config->GetString("3"), "test");
  EXPECT_EQ(sub_config->GetString("in.1"), "");

  sub_config = config->GetSubConfig("graph.node.1");
  EXPECT_EQ(sub_config->Size(), 0);
  EXPECT_EQ(sub_config->GetString("graph.node.1"), "");
  EXPECT_EQ(sub_config->GetString("graph.node.2"), "");

  sub_config = config->GetSubConfig("graph.nothing");
  EXPECT_EQ(sub_config->Size(), 0);
}

TEST_F(ConfigurationTest, BuildFromTomlTest) {
  std::string toml_content = R"(
    [device]
    cpu = "x86"
    freq = "3.5GHZ"
    [device.cpu1]
    cap = 123123
    [device.cpu1.detail]
    vendor=1.3
    sec=false
    [graph]
    data = "123123"
    type = "graphviz"
  )";
  std::string toml_file_path =
      std::string(TEST_DATA_DIR) + "/configure_test.toml";
  std::ofstream ofs(toml_file_path);
  EXPECT_TRUE(ofs.is_open());
  ofs.write(toml_content.data(), toml_content.size());
  ofs.flush();
  ofs.close();
  Defer { remove(toml_file_path.c_str()); };

  ConfigurationBuilder builder;
  auto config = builder.Build(toml_file_path);
  EXPECT_EQ(StatusError, STATUS_SUCCESS);

  EXPECT_EQ(config->Size(), 7);
  EXPECT_EQ(config->GetString("device.cpu"), "x86");
  EXPECT_EQ(config->GetString("device.freq"), "3.5GHZ");
  EXPECT_EQ(config->GetString("device.cpu1.cap"), "123123");
  EXPECT_EQ(config->GetString("device.cpu1.detail.vendor"), "1.3");
  EXPECT_EQ(config->GetString("device.cpu1.detail.sec"), "false");
  EXPECT_EQ(config->GetString("graph.data"), "123123");
  EXPECT_EQ(config->GetString("graph.type"), "graphviz");

  std::set<std::string> expect_value{"cpu", "freq", "cpu1"};
  EXPECT_EQ(config->GetSubKeys("device"), expect_value);

  std::set<std::string> expect_value2{"data", "type"};
  EXPECT_EQ(config->GetSubKeys("graph"), expect_value2);
}

}  // namespace modelbox