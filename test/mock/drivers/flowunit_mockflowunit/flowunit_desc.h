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

#ifndef MODELBOX_FLOWUNIT_DESC_H_
#define MODELBOX_FLOWUNIT_DESC_H_

#include <modelbox/base/device.h>
#include <modelbox/base/driver.h>
#include <modelbox/base/driver_api_helper.h>
#include <modelbox/base/status.h>

#include "flowunit_mockflowunit.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

extern "C" MockDriverFlowUnit *GetDriverMock();

#endif  // MODELBOX_FLOWUNIT_DESC_H_