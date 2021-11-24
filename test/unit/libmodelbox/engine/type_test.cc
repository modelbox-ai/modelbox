
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


#include "modelbox/type.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mockflow.h"

namespace modelbox {

class TypeTest : public testing::Test {
 public:
  TypeTest() {}

 protected:
  virtual void SetUp(){};
  virtual void TearDown(){};
};

#define TEST_TYPE_TypeToDataType(TYPE, DATA_TYPE) \
  {                                               \
    auto type = TypeToDataType<TYPE>::Value;      \
    EXPECT_EQ(type, DATA_TYPE);                   \
  }

TEST_F(TypeTest, TypeToDataType) {
  TEST_TYPE_TypeToDataType(float, MODELBOX_FLOAT);
  TEST_TYPE_TypeToDataType(double, MODELBOX_DOUBLE);
  TEST_TYPE_TypeToDataType(int32_t, MODELBOX_INT32);
  TEST_TYPE_TypeToDataType(uint32_t, MODELBOX_UINT32);
  TEST_TYPE_TypeToDataType(uint16_t, MODELBOX_UINT16);
  TEST_TYPE_TypeToDataType(uint8_t, MODELBOX_UINT8);
  TEST_TYPE_TypeToDataType(int16_t, MODELBOX_INT16);
  TEST_TYPE_TypeToDataType(int8_t, MODELBOX_INT8);
  TEST_TYPE_TypeToDataType(std::string, MODELBOX_STRING);
  TEST_TYPE_TypeToDataType(int64_t, MODELBOX_INT64);
  TEST_TYPE_TypeToDataType(uint64_t, MODELBOX_UINT64);
  TEST_TYPE_TypeToDataType(bool, MODELBOX_BOOL);
}

#define TEST_TYPE_DataTypeSize(TYPE, DATA_TYPE) \
  {                                             \
    auto size = DataTypeSize<DATA_TYPE>::Size;  \
    EXPECT_EQ(size, sizeof(TYPE));              \
  }

TEST_F(TypeTest, DataTypeSize) {
  TEST_TYPE_DataTypeSize(float, MODELBOX_FLOAT);
  TEST_TYPE_DataTypeSize(double, MODELBOX_DOUBLE);
  TEST_TYPE_DataTypeSize(int32_t, MODELBOX_INT32);
  TEST_TYPE_DataTypeSize(uint32_t, MODELBOX_UINT32);
  TEST_TYPE_DataTypeSize(uint16_t, MODELBOX_UINT16);
  TEST_TYPE_DataTypeSize(uint8_t, MODELBOX_UINT8);
  TEST_TYPE_DataTypeSize(int16_t, MODELBOX_INT16);
  TEST_TYPE_DataTypeSize(int8_t, MODELBOX_INT8);
  TEST_TYPE_DataTypeSize(std::string, MODELBOX_STRING);
  TEST_TYPE_DataTypeSize(int64_t, MODELBOX_INT64);
  TEST_TYPE_DataTypeSize(uint64_t, MODELBOX_UINT64);
  TEST_TYPE_DataTypeSize(bool, MODELBOX_BOOL);
}

#define TEST_TYPE_DataTypeToType(TYPE, DATA_TYPE)          \
  {                                                        \
    typedef typename DataTypeToType<DATA_TYPE>::Type type; \
    EXPECT_EQ(typeid(type), typeid(TYPE));                 \
  }

TEST_F(TypeTest, DataTypeToType) {
  TEST_TYPE_DataTypeToType(float, MODELBOX_FLOAT);
  TEST_TYPE_DataTypeToType(double, MODELBOX_DOUBLE);
  TEST_TYPE_DataTypeToType(int32_t, MODELBOX_INT32);
  TEST_TYPE_DataTypeToType(uint32_t, MODELBOX_UINT32);
  TEST_TYPE_DataTypeToType(uint16_t, MODELBOX_UINT16);
  TEST_TYPE_DataTypeToType(uint8_t, MODELBOX_UINT8);
  TEST_TYPE_DataTypeToType(int16_t, MODELBOX_INT16);
  TEST_TYPE_DataTypeToType(int8_t, MODELBOX_INT8);
  TEST_TYPE_DataTypeToType(std::string, MODELBOX_STRING);
  TEST_TYPE_DataTypeToType(int64_t, MODELBOX_INT64);
  TEST_TYPE_DataTypeToType(uint64_t, MODELBOX_UINT64);
  TEST_TYPE_DataTypeToType(bool, MODELBOX_BOOL);
}

}  // namespace modelbox