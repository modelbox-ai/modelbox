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

/* clang-format off */
#include <modelbox/base/log.h>
#include <modelbox/base/utils.h>

#include <list>
#include <toml.hpp>
#include <string.h>

#include <nlohmann/json.hpp>
#include "gtest/gtest.h"
#include "test_config.h"
/* clang-format on */

namespace modelbox {
class BaseUtilsTest : public testing::Test {
 public:
  BaseUtilsTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

TEST_F(BaseUtilsTest, Volume) {
  {
    std::vector<size_t> test({1, 2, 3, 4, 5});
    EXPECT_EQ(Volume(test), 120);
  }

  {
    std::vector<size_t> test({1, 2, 3, 4, 5});
    std::vector<std::vector<size_t>> test_vec;
    for (size_t i = 0; i < 10; ++i) {
      test_vec.push_back(test);
    }

    EXPECT_EQ(Volume(test_vec), 1200);
  }
}

TEST_F(BaseUtilsTest, RegexMatch) {
  const auto *test = "aaa=000 bbb=111   ccc=222     ddd=333";
  EXPECT_TRUE(RegexMatch(test, ".*111.*"));
  EXPECT_TRUE(RegexMatch(test, ".*333$"));
  EXPECT_TRUE(RegexMatch(test, "^aaa.*"));
  EXPECT_FALSE(RegexMatch(test, "^bbb.*"));
}

TEST_F(BaseUtilsTest, StringSplit) {
  const auto *test = "aaa=000 bbb=111   ccc=222     ddd=333";
  auto split_test = StringSplit(test, ' ');
  for (size_t i = 0; i < split_test.size(); ++i) {
    switch (i) {
      case 0:
        EXPECT_EQ(split_test[i], "aaa=000");
        break;
      case 1:
        EXPECT_EQ(split_test[i], "bbb=111");
        break;
      case 2:
        EXPECT_EQ(split_test[i], "ccc=222");
        break;
      case 3:
        EXPECT_EQ(split_test[i], "ddd=333");
        break;
      default:
        break;
    }
  }
}

TEST_F(BaseUtilsTest, BytesReadable) {
  size_t byte = 1;
  size_t kilo = byte * 1024;
  size_t mega = kilo * 1024;
  size_t giga = mega * 1024;
  size_t tera = giga * 1024;
  size_t peta = tera * 1024;
  size_t kilo_half = kilo + kilo / 2;
  size_t mega_half = mega + mega / 2;
  size_t giga_half = giga + giga / 2;
  size_t tera_half = tera + tera / 2;
  size_t peta_half = peta + peta / 2;

  EXPECT_EQ(GetBytesReadable(byte), "1B");
  EXPECT_EQ(GetBytesReadable(kilo), "1KB");
  EXPECT_EQ(GetBytesReadable(mega), "1MB");
  EXPECT_EQ(GetBytesReadable(giga), "1GB");
  EXPECT_EQ(GetBytesReadable(tera), "1TB");
  EXPECT_EQ(GetBytesReadable(peta), "1PB");
  EXPECT_EQ(GetBytesReadable(kilo_half), "1.5KB");
  EXPECT_EQ(GetBytesReadable(mega_half), "1.5MB");
  EXPECT_EQ(GetBytesReadable(giga_half), "1.5GB");
  EXPECT_EQ(GetBytesReadable(tera_half), "1.5TB");
  EXPECT_EQ(GetBytesReadable(peta_half), "1.5PB");
}

TEST_F(BaseUtilsTest, BytesFromReadable) {
  size_t byte = 1;
  size_t kilo = byte * 1024;
  size_t mega = kilo * 1024;
  size_t giga = mega * 1024;
  size_t tera = giga * 1024;
  size_t peta = tera * 1024;
  size_t kilo_half = kilo + kilo / 2;
  size_t mega_half = mega + mega / 2;
  size_t giga_half = giga + giga / 2;
  size_t tera_half = tera + tera / 2;
  size_t peta_half = peta + peta / 2;

  EXPECT_EQ(byte, GetBytesFromReadable("1B"));
  EXPECT_EQ(kilo, GetBytesFromReadable("1KB"));
  EXPECT_EQ(mega, GetBytesFromReadable("1MB"));
  EXPECT_EQ(giga, GetBytesFromReadable("1GB"));
  EXPECT_EQ(tera, GetBytesFromReadable("1TB"));
  EXPECT_EQ(peta, GetBytesFromReadable("1PB"));
  EXPECT_EQ(kilo_half, GetBytesFromReadable("1.5KB"));
  EXPECT_EQ(mega_half, GetBytesFromReadable("1.5MB"));
  EXPECT_EQ(giga_half, GetBytesFromReadable("1.5GB"));
  EXPECT_EQ(tera_half, GetBytesFromReadable("1.5TB"));
  EXPECT_EQ(peta_half, GetBytesFromReadable("1.5PB"));

  EXPECT_EQ(byte, GetBytesFromReadable("1"));
  EXPECT_EQ(kilo, GetBytesFromReadable("1K"));
  EXPECT_EQ(mega, GetBytesFromReadable("1M"));
  EXPECT_EQ(giga, GetBytesFromReadable("1G"));
  EXPECT_EQ(tera, GetBytesFromReadable("1T"));
  EXPECT_EQ(peta, GetBytesFromReadable("1P"));
  EXPECT_EQ(kilo_half, GetBytesFromReadable("1.5K"));
  EXPECT_EQ(mega_half, GetBytesFromReadable("1.5M"));
  EXPECT_EQ(giga_half, GetBytesFromReadable("1.5G"));
  EXPECT_EQ(tera_half, GetBytesFromReadable("1.5T"));
  EXPECT_EQ(peta_half, GetBytesFromReadable("1.5P"));
}

TEST_F(BaseUtilsTest, DeferCondition) {
  int first = 0;
  int second = 0;
  int third = 0;
  int count = 0;
  {
    bool ret = true;
    DeferCond { return ret; };
    DeferCondAdd { first = ++count; };
    DeferCondAdd { second = ++count; };
    DeferCondAdd { third = ++count; };
  }

  EXPECT_EQ(first, 3);
  EXPECT_EQ(second, 2);
  EXPECT_EQ(third, 1);
}

TEST_F(BaseUtilsTest, DeferTest) {
  int i = 0;
  Defer { EXPECT_EQ(i, 3); };
  Defer { i++; };
  Defer { EXPECT_EQ(i, 2); };
  Defer {
    [&]() { i++; }();
  };

  try {
    Defer { i++; };
    throw "exit";
    Defer { i++; };
  } catch (...) {
  }
}

TEST_F(BaseUtilsTest, ListSubDirectoryFiles) {
  std::string filter = "*.toml";
  std::vector<std::string> listfiles;
  std::string python_path = std::string(TEST_ASSETS);
  Status status = ListSubDirectoryFiles(python_path, filter, &listfiles);
  EXPECT_GE(listfiles.size(), 1);
}

TEST_F(BaseUtilsTest, IsAbsolutePath) {
  EXPECT_TRUE(IsAbsolutePath("/"));
  EXPECT_TRUE(IsAbsolutePath(" /"));
  EXPECT_TRUE(IsAbsolutePath(" / "));
  EXPECT_FALSE(IsAbsolutePath("a/"));
  EXPECT_FALSE(IsAbsolutePath("a/ "));
  EXPECT_FALSE(IsAbsolutePath(" a/"));
  EXPECT_FALSE(IsAbsolutePath(" a /"));
  EXPECT_FALSE(IsAbsolutePath("./"));
  EXPECT_FALSE(IsAbsolutePath("../ "));
  EXPECT_FALSE(IsAbsolutePath(" ../"));
  EXPECT_FALSE(IsAbsolutePath(" . /"));
}

TEST_F(BaseUtilsTest, GetDirName) {
  EXPECT_EQ(GetDirName("/"), "/");
  EXPECT_EQ(GetDirName("/a"), "/");
  EXPECT_EQ(GetDirName("/a/"), "/");
  EXPECT_EQ(GetDirName("../"), ".");
  EXPECT_EQ(GetDirName("../../"), "..");
}

TEST_F(BaseUtilsTest, PathCanonicalize) {
  EXPECT_EQ(PathCanonicalize("/"), "/");
  EXPECT_EQ(PathCanonicalize("/../"), "/");
  EXPECT_EQ(PathCanonicalize("/../../"), "/");
  EXPECT_EQ(PathCanonicalize("/a/b/.."), "/a");
  EXPECT_EQ(PathCanonicalize("/a/b/c/../.."), "/a");
  EXPECT_EQ(PathCanonicalize("//a"), "/a");
  EXPECT_EQ(PathCanonicalize("//a/.."), "/");
  EXPECT_EQ(PathCanonicalize("../../", "/a"), "/a");
}

TEST_F(BaseUtilsTest, ListFiles) {
  std::string filter = "*";
  std::vector<std::string> all;
  Status status = ListFiles(TEST_ASSETS, filter, &all, LIST_FILES_ALL);
  EXPECT_TRUE(status);
  std::vector<std::string> alldirs;
  status = ListFiles(TEST_ASSETS, filter, &alldirs, LIST_FILES_DIR);
  EXPECT_TRUE(status);
  std::vector<std::string> allfiles;
  status = ListFiles(TEST_ASSETS, filter, &allfiles, LIST_FILES_FILE);
  EXPECT_TRUE(status);
  EXPECT_EQ(all.size(), alldirs.size() + allfiles.size());
  MBLOG_INFO << all.size();
  MBLOG_INFO << alldirs.size();
  MBLOG_INFO << allfiles.size();
}

TEST_F(BaseUtilsTest, Json2Toml_JsonFailed) {
  std::string jsondata = R"({
    "server": {
        "ip": "0.0.0.0",
    },
    }
  )";

  std::string tomldata;

  bool ret = JsonToToml(jsondata, &tomldata);
  EXPECT_FALSE(ret);
}

TEST_F(BaseUtilsTest, Toml2Json) {
  std::string tomldata = R"(
  root = "root"
  [basic]
  str = "str"
  bool = true
  int = 10
  double = 1.0
  multiline = '''
line1
line2
'''
  array = [
    "array1",
    "array2"
  ]
  [basic.nest]
  aaa = "aaa"
  int = 10
  )";
  std::string json_data;
  auto ret = TomlToJson(tomldata, &json_data);
  std::cout << json_data << std::endl;
  ASSERT_TRUE(ret);
  std::istringstream instring(json_data);
  auto root = nlohmann::json::parse(instring);
  std::cout << root.dump() << std::endl;
  EXPECT_EQ(root["root"], "root");
  EXPECT_EQ(root["basic"]["str"], "str");
  EXPECT_EQ(root["basic"]["bool"], true);
  EXPECT_EQ(root["basic"]["int"], 10);
  EXPECT_EQ(root["basic"]["multiline"], "line1\nline2\n");
  EXPECT_EQ(root["basic"]["nest"]["aaa"], "aaa");
  EXPECT_EQ(root["basic"]["nest"]["int"], 10);
  auto array = root["basic"]["array"];
  EXPECT_EQ(array[0], "array1");
  EXPECT_EQ(array[1], "array2");
}

TEST_F(BaseUtilsTest, Json2Toml) {
  std::string jsondata = R"(
{
    "root": "root",
    "basic": {
        "null": null,
        "str": "str",
        "bool": true,
        "int": 10,
        "double": 1.0,
        "multiline": "a\nb\n",
        "array": [
            "array1",
            "array2"
        ],
        "nest" : {
            "aaa":"aaa",
            "int": 10
        }
    },
    "array": [
        "array1",
        "array2"
    ],
    "nestarray": [
        {
            "nest": "nest1",
            "value": 10,
            "bool": false
        },
        {
            "nest": "nest2"
        }
    ]
}
  )";

  std::string tomldata;

  bool ret = JsonToToml(jsondata, &tomldata);
  ASSERT_TRUE(ret);
  MBLOG_INFO << tomldata;
  std::stringstream ins;
  ins << tomldata;
  auto tom_data = toml::parse(ins);
  EXPECT_EQ(tom_data["root"].as_string(), "root");
  EXPECT_EQ(tom_data["basic"]["str"].as_string(), "str");
  EXPECT_EQ(tom_data["basic"]["bool"].as_boolean(), true);
  EXPECT_EQ(tom_data["basic"]["int"].as_integer(), 10);
  EXPECT_EQ(tom_data["basic"]["double"].as_floating(), 1.0);
  EXPECT_EQ(tom_data["basic"]["multiline"].as_string(), "a\nb\n");
  auto array = tom_data["basic"]["array"].as_array();
  EXPECT_EQ(array[0].as_string(), "array1");
  EXPECT_EQ(array[1].as_string(), "array2");

  auto root_array = tom_data["array"].as_array();
  EXPECT_EQ(root_array[0].as_string(), "array1");
  EXPECT_EQ(root_array[1].as_string(), "array2");

  auto nestarray = tom_data["nestarray"].as_array();
  EXPECT_EQ(nestarray[0]["nest"].as_string(), "nest1");
  EXPECT_EQ(nestarray[1]["nest"].as_string(), "nest2");
}

TEST_F(BaseUtilsTest, FindTheEarliestFileIndex) {
  std::string dir = std::string(TEST_DATA_DIR) + "/test_files";
  auto ret = mkdir(dir.c_str(), 0700);
  EXPECT_EQ(ret, 0);
  for (int i = 1; i <= 5; ++i) {
    auto open_file = dir + "/" + std::to_string(i) + ".txt";
    std::ofstream out(open_file, std::ios::binary | std::ios::trunc);
    EXPECT_EQ(out.fail(), false);
    out << i;
    out.close();
  }

  std::vector<std::string> list_files;
  auto status = ListSubDirectoryFiles(dir, "*.txt", &list_files);
  EXPECT_EQ(status, modelbox::STATUS_OK);
  auto earliest_file_index = FindTheEarliestFileIndex(list_files);
  EXPECT_EQ(earliest_file_index, 0);
  EXPECT_EQ(list_files.size(), 5);

  for (const auto& file : list_files) {
    auto ret = remove(file.c_str());
    EXPECT_EQ(ret, 0);
  }
  ret = remove(dir.c_str());
  EXPECT_EQ(ret, 0);
}

TEST_F(BaseUtilsTest, StrError) {
  int err = EACCES;
  MBLOG_INFO << modelbox::StrError(err);
  EXPECT_EQ(modelbox::StrError(err), strerror(err));
  err = 245;
  MBLOG_INFO << modelbox::StrError(err);
  EXPECT_EQ(modelbox::StrError(err), strerror(err));
}

}  // namespace modelbox
