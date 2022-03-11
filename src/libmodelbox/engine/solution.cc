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

#include <modelbox/flow.h>
#include <modelbox/solution.h>

namespace modelbox {
Solution::Solution(const std::string &solution_name) {
  solution_name_ = solution_name;
}

Solution::~Solution() {}

void Solution::SetSolutionDir(const std::string &dir) { solution_dir_ = dir; }

Solution &Solution::SetArgs(const std::string &key, const std::string &value) {
  args_.emplace(key, value);
  return *this;
}

const std::string Solution::GetSolutionDir() const { return solution_dir_; }

const std::string Solution::GetSolutionName() const { return solution_name_; }

}  // namespace modelbox
