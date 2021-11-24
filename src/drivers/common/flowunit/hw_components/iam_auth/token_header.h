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

#ifndef TOKEN_HEADER_H_
#define TOKEN_HEADER_H_
#include <iostream>

namespace modelbox {
class AccountInfo {
 public:
  std::string domain_name;
  std::string user_name;
  std::string passwd;
};

class ConsigneeInfo {
 public:
  std::string ak;
  std::string sk;
  std::string domain_id;
  std::string project_id;
};

class UserAgencyCredential {
 public:
  std::string user_id;
  std::string user_ak;
  std::string user_sk;
  std::string user_secure_token;
};


class AgentToken {
 public:
  std::string expires_time_;
  std::string x_subject_token_;
};

class UserAgencyToken {
 public:
  std::string user_token;
};

class AgencyInfo {
 public:
  std::string user_domain_name;
  std::string xrole_name;

  bool operator<(const AgencyInfo &agency_info) const {
    if (this->user_domain_name < agency_info.user_domain_name) {
      return true;
    }
    if ((this->user_domain_name == agency_info.user_domain_name) &&
        (this->xrole_name < agency_info.xrole_name)) {
      return true;
    }
    return false;
  }
};

class ProjectInfo {
 public:
  std::string project_name;
  std::string project_id;

  bool operator<(const ProjectInfo &project_info) const {
    if (this->project_name < project_info.project_name) {
      return true;
    }
    if ((this->project_name == project_info.project_name) &&
        (this->project_id < project_info.project_id)) {
      return true;
    }
    return false;
  }
};

}  // namespace modelbox
#endif