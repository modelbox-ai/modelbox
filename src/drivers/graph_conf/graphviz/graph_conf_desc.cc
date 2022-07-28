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

#include <stdio.h>

#include <memory>

#include "graphviz_conf.h"
#include "modelbox/base/driver_api_helper.h"
#include "modelbox/base/status.h"

std::shared_ptr<modelbox::DriverFactory> CreateDriverFactory() {
  std::shared_ptr<modelbox::DriverFactory> factory =
      std::make_shared<modelbox::GraphvizFactory>();
  return factory;
}

void DriverDescription(modelbox::DriverDesc *desc) {
  desc->SetName(modelbox::GRAPHCONF_NAME);
  desc->SetClass(modelbox::DRIVER_CLASS_GRAPHCONF);
  desc->SetType(modelbox::GRAPHCONF_TYPE);
  desc->SetVersion(modelbox::GRAPHVIZE_VERSION);
  desc->SetDescription(modelbox::GRAPHCONF_DESC);
}

modelbox::Status DriverInit() { return modelbox::STATUS_OK; }

void DriverFini() {}
