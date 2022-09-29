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

#include "modelbox/base/driver.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "modelbox/base/config.h"
#include "modelbox/base/driver_utils.h"
#include "modelbox/base/log.h"

namespace modelbox {

constexpr const char *DEFAULT_LD_CACHE = "/etc/ld.so.cache";
constexpr const int DRIVER_SCAN_TIMEOUT = 60 * 3;

int SubProcessWaitAndLog(int fd) {
  struct pollfd fdset;
  std::string log;
  char tmp[4096];
  fdset.fd = fd;
  fdset.events = POLLIN | POLLHUP;
  time_t begin = 0;
  time_t now;
  if (fd <= 0) {
    return 0;
  }

  Defer {
    if (log.length() > 0) {
      MBLOG_INFO << "scan process log: \n" << log;
    }
  };

  fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);

  time(&begin);
  while (true) {
    int count = poll(&fdset, 1, 10000);
    time(&now);

    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }

      return -1;
    }

    if (count == 0) {
      if (now - begin >= DRIVER_SCAN_TIMEOUT) {
        return 1;
      }

      continue;
    }

    int len = read(fd, tmp, sizeof(tmp));
    if (len < 0) {
      return -1;
    }

    if (len == 0) {
      break;
    }

    log.append(tmp, len);
    if (log.length() > 4096) {
      MBLOG_INFO << "scan process log: \n" << log;
      log.clear();
    }
  }

  return 0;
}

/**
 * @brief fork a process to Run func
 * @return func result
 */
template <typename func, typename... ts>
Status SubProcessRun(func &&fun, ts &&...params) {
  const char *enable_debug = getenv("MODELBOX_DEBUG_DRIVER_SCAN");
  if (enable_debug == nullptr) {
    int unused __attribute__((unused));
    int fd[2] = {-1, -1};
    unused = pipe(fd);
    auto pid = fork();
    if (pid == 0) {
      signal(SIGSEGV, SIG_DFL);
      signal(SIGTERM, SIG_DFL);
      signal(SIGABRT, SIG_DFL);
      close(fd[0]);
      dup2(fd[1], 1);
      close(fd[1]);
      setbuf(stdout, nullptr);

      // Keep old log avoid deadlock
      auto oldlogger_keeper = klogger.GetLogger();

      // output log to console
      klogger.SetLogger(nullptr);
      klogger.GetLogger()->SetLogLevel(oldlogger_keeper->GetLogLevel());
      Status ret = fun(params...);
      if (ret == STATUS_OK) {
        _exit(0);
      }

      MBLOG_WARN << "run function failed, errmsg: " << ret.WrapErrormsgs();
      _exit(1);
    }

    Defer { close(fd[0]); };
    close(fd[1]);

    if (pid == -1) {
      const auto *err_msg = "fork subprocess failed";
      MBLOG_ERROR << err_msg;
      return {STATUS_NOMEM, err_msg};
    }

    MBLOG_INFO << "wait for subprocess " << pid << " process finished";
    int status = 0;

    auto ret = SubProcessWaitAndLog(fd[0]);
    if (ret == 1) {
      MBLOG_WARN << "scan process timeout, kill scan process.";
      kill(pid, 9);
    }

    ret = waitpid(pid, &status, 0);
    if (ret < 0) {
      auto err_msg =
          "subprocess run failed, wait error, ret:" + std::to_string(errno) +
          ", msg: " + StrError(errno);
      MBLOG_ERROR << err_msg;
      return {STATUS_FAULT, err_msg};
    }

    if (WIFSIGNALED(status)) {
      const auto *err_msg = "killed by signal";
      MBLOG_ERROR << err_msg;
      return {STATUS_NORESPONSE, err_msg};
    }

    if (!WIFEXITED(status)) {
      std::string err_msg =
          "process exit abnormal, ret = " + std::to_string(status);
      MBLOG_ERROR << err_msg;
      return {STATUS_NORESPONSE, err_msg};
    }

    if (status != 0) {
      const auto *err_msg = "scan process exit result is fail.";
      MBLOG_ERROR << err_msg;
      return {STATUS_FAULT, err_msg};
    }
  } else {
    return fun(params...);
  }

  return STATUS_OK;
}

Driver::Driver() = default;

Driver::~Driver() {
  if (factory_count_ != 0) {
    Abort("factory reference count is not zero");
  }
}

std::string Drivers::default_scan_path_ = MODELBOX_DEFAULT_DRIVER_PATH;
std::string Drivers::default_driver_info_path_ = DEFAULT_SCAN_INFO;

static std::shared_ptr<DriverHandler> handler =
    std::make_shared<DriverHandler>();
// Driver
std::string Driver::GetDriverFile() { return GetDriverDesc()->GetFilePath(); }

bool Driver::IsVirtual() { return is_virtual_; }

void Driver::SetVirtual(bool is_virtual) { is_virtual_ = is_virtual; }

void Driver::CloseFactory() {
  std::lock_guard<std::mutex> guard(mutex_);
  CloseFactoryLocked();
}

void Driver::CloseFactoryLocked() {
  factory_count_--;
  if (factory_count_ > 0) {
    return;
  }

  if (!driver_handler_) {
    factory_ = nullptr;
    return;
  }

  handler->handler_map_lock.lock();
  auto driver_handler_info = handler->Get(driver_handler_);
  if (driver_handler_info == nullptr) {
    MBLOG_ERROR << "close factory failed, get null driver_handler_info";
  }
  handler->handler_map_lock.unlock();

  auto no_delete = GetDriverDesc()->GetNoDelete();
  driver_handler_info->initialize_lock_.lock();
  if (--driver_handler_info->initialize_count_ == 0) {
    if (!no_delete) {
      typedef void (*DriverFini)();
      DriverFini driver_fini = nullptr;
      driver_fini = (DriverFini)dlsym(driver_handler_, "DriverFini");
      if (driver_fini) {
        driver_fini();
      }

      handler->handler_map_lock.lock();
      handler->Remove(driver_handler_);
      handler->handler_map_lock.unlock();
    } else {
      driver_handler_info->initialize_count_++;
    }
  }

  driver_handler_info->initialize_lock_.unlock();
  factory_ = nullptr;
  dlclose(driver_handler_);
  driver_handler_ = nullptr;
}

std::shared_ptr<DriverHandlerInfo> DriverHandler::Add(void *driver_handler) {
  auto driver_handler_info = Get(driver_handler);
  if (nullptr == driver_handler_info) {
    std::shared_ptr<DriverHandlerInfo> driver_handler_info =
        std::make_shared<DriverHandlerInfo>();
    driver_handler_info->IncHanderRefcnt();
    handler_map.emplace(driver_handler, driver_handler_info);
    return driver_handler_info;
  }

  driver_handler_info->IncHanderRefcnt();
  return driver_handler_info;
}

Status DriverHandler::Remove(void *driver_handler) {
  auto driver_handler_info = Get(driver_handler);
  auto cnt = driver_handler_info->DecHanderRefcnt();
  if (cnt == 0) {
    handler_map.erase(driver_handler);
  }
  return STATUS_SUCCESS;
}

std::shared_ptr<DriverHandlerInfo> DriverHandler::Get(void *driver_handler) {
  auto driver_handler_info = handler_map.find(driver_handler);
  if (driver_handler_info == handler_map.end()) {
    return nullptr;
  }

  return driver_handler_info->second;
}

int Driver::GetMode(bool no_delete, bool global, bool deep_bind) {
  unsigned int mode = RTLD_NOW;
  if (no_delete) {
    mode |= RTLD_NODELETE;
  }

#ifdef RTLD_DEEPBIND
  if (deep_bind) {
    mode |= RTLD_DEEPBIND;
  }
#endif

  if (global) {
    mode |= RTLD_GLOBAL;
    return mode;
  }

  mode |= RTLD_LOCAL;
  return static_cast<int>(mode);
}

std::shared_ptr<DriverFactory> Driver::CreateFactory() {
  std::lock_guard<std::mutex> guard(mutex_);
  factory_count_++;
  if (factory_count_ == 1) {
    auto no_delete = GetDriverDesc()->GetNoDelete();
    auto global = GetDriverDesc()->GetGlobal();
    auto deep_bind = GetDriverDesc()->GetDeepBind();
    typedef std::shared_ptr<DriverFactory> (*CreateDriverFactory)();
    typedef Status (*DriverInit)();

    CreateDriverFactory driver_func = nullptr;
    DriverInit driver_init = nullptr;

    int mode = GetMode(no_delete, global, deep_bind);

    driver_handler_ = dlopen(GetDriverFile().c_str(), mode);
    if (driver_handler_ == nullptr) {
      const char *dl_errmsg = dlerror();
      if (dl_errmsg == nullptr) {
        dl_errmsg = "no error msg";
      }
      StatusError = {STATUS_INVALID, "dlopen " + GetDriverFile() +
                                         " failed, error: " + dl_errmsg};
      MBLOG_ERROR << StatusError.Errormsg();
      CloseFactoryLocked();
      return nullptr;
    }

    handler->handler_map_lock.lock();
    auto handler_info = handler->Add(driver_handler_);
    handler->handler_map_lock.unlock();

    handler_info->initialize_lock_.lock();
    if (++handler_info->initialize_count_ == 1) {
      driver_init = (DriverInit)dlsym(driver_handler_, "DriverInit");
      if (driver_init == nullptr) {
        handler_info->initialize_count_--;
        handler_info->initialize_lock_.unlock();
        const char *dl_errmsg = dlerror();
        if (dl_errmsg == nullptr) {
          dl_errmsg = "no error msg";
        }
        StatusError = {STATUS_INVALID,
                       "failed to dlsym function DriverInit in file: " +
                           GetDriverFile() + ", error: " + dl_errmsg};
        CloseFactoryLocked();
        return nullptr;
      }

      Status init = driver_init();
      if (init != STATUS_OK) {
        handler_info->initialize_count_--;
        handler_info->initialize_lock_.unlock();
        StatusError = {init, "driver init failed, driver:" + GetDriverFile()};
        MBLOG_ERROR << "driverInit failed in " << GetDriverFile() << ", "
                    << init;
        CloseFactoryLocked();
        return nullptr;
      }
    }

    handler_info->initialize_lock_.unlock();

    driver_func =
        (CreateDriverFactory)dlsym(driver_handler_, "CreateDriverFactory");
    if (driver_func == nullptr) {
      auto *dl_err_msg = dlerror();
      if (dl_err_msg != nullptr) {
        StatusError = {STATUS_INVALID,
                       "failed to dlsym function DriverDescription in file: " +
                           GetDriverFile() + ", error: " + dl_err_msg};
      } else {
        StatusError = {STATUS_INVALID,
                       "failed to dlsym function DriverDescription in file: " +
                           GetDriverFile() + ", error: no error msg."};
      }

      MBLOG_ERROR << StatusError.Errormsg();
      CloseFactoryLocked();
      return nullptr;
    }

    factory_ = driver_func();
    if (!factory_) {
      StatusError = {STATUS_FAULT,
                     "create driver failed, driver:" + GetDriverFile()};
      MBLOG_ERROR << StatusError.Errormsg();
      CloseFactoryLocked();
      return nullptr;
    }
  }

  auto holder = shared_from_this();
  std::shared_ptr<DriverFactory> child_factory(
      factory_.get(),
      [&, holder](DriverFactory *child_factory) { holder->CloseFactory(); });

  return child_factory;
}

std::shared_ptr<DriverDesc> Driver::GetDriverDesc() { return desc_; }

void Driver::SetDriverDesc(std::shared_ptr<DriverDesc> desc) {
  desc_ = std::move(desc);
}

DriverFactory::DriverFactory() = default;
DriverFactory::~DriverFactory() = default;

std::shared_ptr<Driver> DriverFactory::GetDriver() {
  return std::make_shared<Driver>();
};

void DriverFactory::SetDriver(const std::shared_ptr<Driver> &driver) {}

// DriverDesc
std::string DriverDesc::GetClass() { return driver_class_; }

std::string DriverDesc::GetType() { return driver_type_; }

std::string DriverDesc::GetName() { return driver_name_; }

std::string DriverDesc::GetDescription() { return driver_description_; }

std::string DriverDesc::GetVersion() { return driver_version_; }

std::string DriverDesc::GetFilePath() { return driver_file_path_; }

bool DriverDesc::GetNoDelete() { return driver_no_delete_; }

bool DriverDesc::GetGlobal() { return global_; }
bool DriverDesc::GetDeepBind() { return deep_bind_; }

void DriverDesc::SetClass(const std::string &classname) {
  driver_class_ = classname;
}

void DriverDesc::SetType(const std::string &type) { driver_type_ = type; }

void DriverDesc::SetName(const std::string &name) { driver_name_ = name; }

void DriverDesc::SetDescription(const std::string &description) {
  driver_description_ = description;
}

void DriverDesc::SetNodelete(const bool &no_delete) {
  driver_no_delete_ = no_delete;
}

void DriverDesc::SetGlobal(const bool &global) { global_ = global; }
void DriverDesc::SetDeepBind(const bool &deep_bind) { deep_bind_ = deep_bind; }

Status DriverDesc::SetVersion(const std::string &version) {
  if (version.empty()) {
    return STATUS_SUCCESS;
  }

  Status status = CheckVersion(version);
  if (status != STATUS_SUCCESS) {
    MBLOG_ERROR << "SetVersion failed, the version model is: x.y.z (xyz should "
                   "be integer), version is: "
                << version;
    return status;
  }

  driver_version_ = version;
  return STATUS_SUCCESS;
}

Status DriverDesc::CheckVersion(const std::string &version) {
  std::vector<std::string> version_;
  std::istringstream iss(version);
  std::string temp;
  char split_char = '.';

  if (version.find(split_char) == std::string::npos) {
    return {STATUS_BADCONF, "version is invalid"};
  }

  while (std::getline(iss, temp, split_char)) {
    version_.emplace_back(std::move(temp));
  }

  if (version_.size() != 3) {
    // "x", "y", "z"
    return {STATUS_BADCONF, "version is invalid"};
  }

  for (auto &item_version : version_) {
    if (std::all_of(item_version.begin(), item_version.end(), ::isdigit)) {
      continue;
    }
    return {STATUS_BADCONF, "version is invalid"};
  }

  return STATUS_SUCCESS;
}

void DriverDesc::SetFilePath(const std::string &file_path) {
  driver_file_path_ = file_path;
}

DriversScanResultInfo::DriversScanResultInfo() = default;
DriversScanResultInfo::~DriversScanResultInfo() {
  load_success_info_.clear();
  load_failed_info_.clear();
}

std::list<std::string> &DriversScanResultInfo::GetLoadSuccessInfo() {
  return load_success_info_;
}

std::map<std::string, std::string> &DriversScanResultInfo::GetLoadFailedInfo() {
  return load_failed_info_;
}

// Drivers
Drivers::Drivers()
    : drivers_scan_result_info_(std::make_shared<DriversScanResultInfo>()){};
Drivers::~Drivers() = default;

std::shared_ptr<Drivers> Drivers::GetInstance() {
  static std::shared_ptr<Drivers> drivers = std::make_shared<Drivers>();
  return drivers;
}

void Drivers::PrintScanResult(
    const std::list<std::string> &load_success_info,
    const std::map<std::string, std::string> &load_failed_info) {
  if (load_success_info.empty()) {
    MBLOG_WARN << "no driver load success, please check";
  } else {
    MBLOG_INFO << "load success drivers: count " << load_success_info.size()
               << ", show detail in debug level";
    for (const auto &info : load_success_info) {
      MBLOG_DEBUG << info;
    }
  }

  if (load_failed_info.empty()) {
    MBLOG_INFO << "no drivers load failed";
  } else {
    MBLOG_WARN << "load failed drivers: count " << load_failed_info.size()
               << ", detail:";
    for (const auto &info : load_failed_info) {
      MBLOG_WARN << info.second;
    }
  }
}

Status Drivers::Scan(const std::string &path, const std::string &filter) {
  std::vector<std::string> drivers_list;
  struct stat s;
  auto ret = lstat(path.c_str(), &s);
  if (ret) {
    auto err_msg = "lstat " + path + " failed, " + StrError(errno);
    return {STATUS_FAULT, err_msg};
  }

  if (!S_ISDIR(s.st_mode)) {
    last_modify_time_sum_ += s.st_mtim.tv_sec;
    auto status = Add(path);
    if (status == STATUS_OK) {
      drivers_scan_result_info_->GetLoadSuccessInfo().push_back(path);
    } else {
      drivers_scan_result_info_->GetLoadFailedInfo().emplace(path,
                                                             status.Errormsg());
    }
    return status;
  }

  Status status = ListFiles(path, filter, &drivers_list);
  if (status != STATUS_OK) {
    auto err_msg = "list directory:  " + path + "/" + filter + " failed, ";
    return {status, err_msg};
  }

  if (drivers_list.size() == 0) {
    return {STATUS_NOTFOUND, "directory is empty"};
  }

  for (auto &driver_file : drivers_list) {
    struct stat buf;
    auto ret = lstat(driver_file.c_str(), &buf);
    if (ret) {
      continue;
    }

    if (S_ISLNK(buf.st_mode)) {
      continue;
    }
    last_modify_time_sum_ += buf.st_mtim.tv_sec;

    auto result = Add(driver_file);
    if (result == STATUS_OK) {
      drivers_scan_result_info_->GetLoadSuccessInfo().push_back(driver_file);
    } else {
      drivers_scan_result_info_->GetLoadFailedInfo().emplace(driver_file,
                                                             result.Errormsg());
    }
  }

  return STATUS_OK;
}

Status Drivers::Initialize(const std::shared_ptr<Configuration> &config) {
  if (config == nullptr) {
    return {STATUS_INVALID, "config is empty."};
  }
  config_ = config;

  driver_dirs_ = config_->GetStrings(DRIVER_DIR);
  if (config_->GetBool(DRIVER_SKIP_DEFAULT, false) == false) {
    driver_dirs_.push_back(default_scan_path_);
  }

  MBLOG_DEBUG << "search Path:";
  for (const auto &dir : driver_dirs_) {
    MBLOG_DEBUG << " " << dir;
  }

  return STATUS_OK;
}

Status Drivers::WriteScanInfo(const std::string &scan_info_path,
                              const std::string &check_code) {
  std::ofstream scan_info_file(scan_info_path);
  if (!scan_info_file.is_open()) {
    return {STATUS_FAULT, "Open file " + scan_info_path + " failed, " + StrError(errno)};
  }

  nlohmann::json dump_json;

  struct stat buffer;
  if (stat(DEFAULT_LD_CACHE, &buffer) == -1) {
    dump_json["ld_cache_time"] = 0;
  } else {
    dump_json["ld_cache_time"] = buffer.st_mtim.tv_sec;
  }

  dump_json["check_code"] = check_code;
  std::time_t tt =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  dump_json["version_record"] = std::ctime(&tt);
  nlohmann::json dump_driver_json_arr = nlohmann::json::array();

  MBLOG_DEBUG << "write info begin";
  for (auto &driver : drivers_list_) {
    nlohmann::json dump_driver_json;
    auto desc = driver->GetDriverDesc();
    auto cls = desc->GetClass();
    auto type = desc->GetType();
    auto name = desc->GetName();
    auto description = desc->GetDescription();
    auto version = desc->GetVersion();
    auto file_path = desc->GetFilePath();
    auto no_delete = desc->GetNoDelete();
    auto global = desc->GetGlobal();
    auto deep_bind = desc->GetDeepBind();
    dump_driver_json["class"] = cls;
    dump_driver_json["type"] = type;
    dump_driver_json["name"] = name;
    dump_driver_json["description"] = description;
    dump_driver_json["version"] = version;
    dump_driver_json["file_path"] = file_path;
    dump_driver_json["no_delete"] = no_delete;
    dump_driver_json["global"] = global;
    dump_driver_json["deep_bind"] = deep_bind;
    dump_driver_json["load_success"] = true;
    dump_driver_json_arr.push_back(dump_driver_json);
  }

  auto load_failed_info = drivers_scan_result_info_->GetLoadFailedInfo();
  MBLOG_DEBUG << "load_failed_info size " << load_failed_info.size();
  for (auto &fail_info : load_failed_info) {
    nlohmann::json dump_driver_json;
    dump_driver_json["file_path"] = fail_info.first;
    dump_driver_json["err_msg"] = fail_info.second;
    dump_driver_json["load_success"] = false;
    dump_driver_json_arr.push_back(dump_driver_json);
  }

  dump_json["scan_drivers"] = dump_driver_json_arr;
  scan_info_file << dump_json;
  MBLOG_DEBUG << "write info end";

  scan_info_file.close();
  return STATUS_OK;
}

Status Drivers::GatherScanInfo(const std::string &scan_path) {
  std::ifstream scan_info_file(scan_path);
  if (!scan_info_file.is_open()) {
    MBLOG_ERROR << "Open file " << scan_path << " for read failed";
    return STATUS_FAULT;
  }

  nlohmann::json dump_json;
  try {
    std::string ss((std::istreambuf_iterator<char>(scan_info_file)),
                   std::istreambuf_iterator<char>());
    dump_json = nlohmann::json::parse(ss);

    auto driver_json_arr = dump_json["scan_drivers"];
    for (auto &driver_info : driver_json_arr) {
      if (!driver_info["load_success"]) {
        continue;
      }

      auto driver = std::make_shared<Driver>();
      auto desc = driver->GetDriverDesc();
      desc->SetClass(driver_info["class"]);
      desc->SetType(driver_info["type"]);
      desc->SetName(driver_info["name"]);
      desc->SetDescription(driver_info["description"]);
      desc->SetVersion(driver_info["version"]);
      desc->SetFilePath(driver_info["file_path"]);
      desc->SetNodelete(driver_info["no_delete"]);
      desc->SetGlobal(driver_info["global"]);
      desc->SetDeepBind(driver_info["deep_bind"]);
      auto tmp_driver = GetDriver(driver_info["class"], driver_info["type"],
                                  driver_info["name"], driver_info["version"]);
      if (tmp_driver == nullptr) {
        drivers_list_.push_back(driver);
      }
    }
  } catch (const std::exception &e) {
    auto err_msg = "gather scan info failed, err: " + std::string(e.what());
    MBLOG_ERROR << err_msg;
    return {STATUS_FAULT, err_msg};
  }

  MBLOG_INFO << "Gather scan info success, drivers count "
             << drivers_list_.size();
  return STATUS_OK;
}

Status Drivers::FillCheckInfo(std::string &file_check_node,
                              std::unordered_map<std::string, bool> &file_map,
                              int64_t &ld_cache_time) {
  std::ifstream scan_info(default_driver_info_path_);
  if (!scan_info.is_open()) {
    MBLOG_ERROR << "open " << default_driver_info_path_ << " failed.";
    return {STATUS_FAULT, "scan info file is not found"};
  }

  nlohmann::json dump_json;
  try {
    std::string ss((std::istreambuf_iterator<char>(scan_info)),
                   std::istreambuf_iterator<char>());
    dump_json = nlohmann::json::parse(ss);

    file_check_node = dump_json["check_code"];
    ld_cache_time = dump_json["ld_cache_time"];
    auto driver_json_arr = dump_json["scan_drivers"];
    for (const auto &driver_info : driver_json_arr) {
      if (file_map.find(driver_info["file_path"]) != file_map.end()) {
        continue;
      }
      file_map[driver_info["file_path"]] = true;
    }
  } catch (const std::exception &e) {
    MBLOG_WARN << "filee check info parse " << default_driver_info_path_
               << " failed, err: " << e.what();
    return {STATUS_INVALID,
            std::string("parser scan info file failed, ") + e.what()};
  }

  return STATUS_SUCCESS;
}

bool Drivers::CheckPathAndMagicCode() {
  struct stat buffer;
  if (stat(default_driver_info_path_.c_str(), &buffer) == -1) {
    MBLOG_DEBUG << default_driver_info_path_ << " does not exist.";
    return false;
  }

  if (stat(DEFAULT_LD_CACHE, &buffer) == -1) {
    MBLOG_DEBUG << DEFAULT_LD_CACHE << " does not exit.";
    return false;
  }

  std::string file_check_node;
  std::unordered_map<std::string, bool> file_map;
  int64_t ld_cache_time = 0;
  auto ret = FillCheckInfo(file_check_node, file_map, ld_cache_time);
  if (ret != STATUS_SUCCESS) {
    MBLOG_DEBUG << "get check info failed, file: " << default_driver_info_path_
                << " error:" << ret.Errormsg();
    return false;
  }

  if (ld_cache_time != buffer.st_mtim.tv_sec) {
    return false;
  }

  int64_t check_sum = 0;
  for (const auto &dir : driver_dirs_) {
    std::vector<std::string> drivers_list;
    std::string filter = "libmodelbox-*.so*";
    struct stat s;
    auto ret = lstat(dir.c_str(), &s);
    if (ret) {
      MBLOG_ERROR << "lstat " << dir << " failed, errno:" << StrError(errno);
      return false;
    }

    if (!S_ISDIR(s.st_mode)) {
      check_sum += s.st_mtim.tv_sec;
      continue;
    }

    Status status = ListFiles(dir, filter, &drivers_list);
    if (status != STATUS_OK) {
      auto err_msg = "list directory:  " + dir + "/";
      err_msg += filter + " failed, ";
      if (status != STATUS_NOTFOUND) {
        MBLOG_ERROR << err_msg << status.WrapErrormsgs();
      }
      return false;
    }

    if (drivers_list.size() == 0) {
      continue;
    }

    for (auto &driver_file : drivers_list) {
      struct stat buf;
      auto ret = lstat(driver_file.c_str(), &buf);
      if (ret) {
        MBLOG_DEBUG << "lstat " << dir << " failed, errno:" << StrError(errno);
        continue;
      }

      if (S_ISLNK(buf.st_mode)) {
        continue;
      }

      if (file_map.count(driver_file) == 0) {
        return false;
      }

      check_sum += buf.st_mtim.tv_sec;
    }
  }
  auto check_code = GenerateKey(check_sum);
  if (file_check_node != check_code) {
    return false;
  }

  return true;
}

Status Drivers::InnerScan() {
  Status ret = STATUS_NOTFOUND;
  for (const auto &dir : driver_dirs_) {
    MBLOG_INFO << "Scan dir: " << dir;
    ret = Scan(dir, "libmodelbox-*.so*");
    if (!ret && ret != STATUS_NOTFOUND) {
      MBLOG_WARN << "scan " << dir << " failed, " << ret.WrapErrormsgs();
    }
    ret = STATUS_OK;
  }

  auto check_code = GenerateKey(last_modify_time_sum_);

  ret = WriteScanInfo(default_driver_info_path_, check_code);
  if (ret != STATUS_OK) {
    std::string err_msg = "write scan info failed, " + ret.WrapErrormsgs();
    MBLOG_ERROR << err_msg;
    return {ret, err_msg};
  }

  return ret;
}

void Drivers::PrintScanResults(const std::string &scan_path) {
  std::ifstream scan_info_file(scan_path);
  if (!scan_info_file.is_open()) {
    MBLOG_ERROR << "Open file " << scan_path << " for read failed";
    return;
  }

  nlohmann::json dump_json;
  try {
    std::string ss((std::istreambuf_iterator<char>(scan_info_file)),
                   std::istreambuf_iterator<char>());
    dump_json = nlohmann::json::parse(ss);

    nlohmann::json dump_driver_json_arr = nlohmann::json::array();
    dump_driver_json_arr = dump_json["scan_drivers"];

    std::list<std::string> load_success_info;
    std::map<std::string, std::string> load_failed_info;

    for (auto &dump_json : dump_driver_json_arr) {
      if (dump_json["load_success"]) {
        load_success_info.push_back(dump_json["file_path"]);
        continue;
      }

      load_failed_info.emplace(dump_json["file_path"], dump_json["err_msg"]);
    }

    PrintScanResult(load_success_info, load_failed_info);

  } catch (const std::exception &e) {
    MBLOG_ERROR << "print scan result failed, err: " << e.what();
    return;
  }
}

Status Drivers::ReadExcludeInfo() {
  std::ifstream infile(scan_info_file_);
  if (infile.fail()) {
    std::string msg = "read scan info file " + scan_info_file_ + " failed, " +
                      StrError(errno);
    return {STATUS_IO, msg};
  }

  Defer { infile.close(); };
  std::string data((std::istreambuf_iterator<char>(infile)),
                   std::istreambuf_iterator<char>());

  if (access(data.c_str(), F_OK) != 0) {
    return {STATUS_NOENT, "not found info file" + data};
  }

  scan_exclude_file_list_[data] = true;

  return STATUS_OK;
}

Status Drivers::Scan() {
  Status status = STATUS_FAULT;
  int retry_count = 0;
  if (!CheckPathAndMagicCode()) {
    while (true) {
      scan_info_file_ = DRIVER_SCAN_INFO;
      Defer { unlink(scan_info_file_.c_str()); };
      auto exec_func = std::bind(&Drivers::InnerScan, this);
      auto status = SubProcessRun(exec_func);

      if (status == STATUS_NORESPONSE) {
        MBLOG_WARN << "Scan process may crash, retry.";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (ReadExcludeInfo() != STATUS_OK) {
          retry_count++;
          if (retry_count < 3) {
            continue;
          }

          break;
        }

        continue;
      }

      scan_exclude_file_list_.clear();
      if (status != STATUS_OK) {
        auto err_msg =
            "fork subprocess run scan so failed, " + status.WrapErrormsgs();
        MBLOG_ERROR << err_msg;
        return {STATUS_FAULT, err_msg};
      }

      break;
    }
  }

  status = GatherScanInfo(default_driver_info_path_);
  if (status != STATUS_OK) {
    const auto *err_msg = "gather scan info failed";
    MBLOG_ERROR << err_msg;
    return {STATUS_FAULT, err_msg};
  }

  MBLOG_INFO << "begin scan virtual drivers";
  status = VirtualDriverScan();
  MBLOG_INFO << "end scan virtual drivers";

  return status;
}

void Drivers::Clear() {
  for (auto iter = drivers_list_.begin(); iter != drivers_list_.end();) {
    if (iter->get()->IsVirtual() != true) {
      iter++;
      continue;
    }

    drivers_list_.erase(iter);
  }
  virtual_driver_manager_list_.clear();
  drivers_list_.clear();
  driver_dirs_.clear();
  config_ = nullptr;
  last_modify_time_sum_ = 0;
}

Status Drivers::VirtualDriverScan() {
  for (auto &driver : GetDriverListByClass(DRIVER_CLASS_VIRTUAL)) {
    std::shared_ptr<VirtualDriverManager> factory =
        std::dynamic_pointer_cast<VirtualDriverManager>(
            driver->CreateFactory());

    if (factory == nullptr) {
      continue;
    }

    auto result = factory->Init(*this);
    if (result != STATUS_SUCCESS) {
      MBLOG_WARN << "virtual driver init failed, " << result;
    }

    result = factory->Scan(driver_dirs_);
    if (result != STATUS_SUCCESS) {
      MBLOG_WARN << "scan failed, " << result;
    }

    for (const auto &virtualDriver : factory->GetAllDriverList()) {
      drivers_list_.push_back(virtualDriver);
    }

    virtual_driver_manager_list_.push_back(factory);
  }

  return STATUS_OK;
}

Status Drivers::Add(const std::string &file) {
  typedef void (*DriverDescription)(DriverDesc *);
  DriverDescription driver_func = nullptr;

  if (scan_exclude_file_list_.find(file) != scan_exclude_file_list_.end()) {
    MBLOG_WARN << "Skip scan file: " << file;
    return STATUS_OK;
  }

  if (!scan_info_file_.empty()) {
    std::ofstream out(scan_info_file_, std::ios::trunc);
    if (!out.fail()) {
      chmod(scan_info_file_.c_str(), 0600);
      Defer { out.close(); };
      out << file;
    }
  }

  Defer {
    if (!scan_info_file_.empty()) {
      unlink(scan_info_file_.c_str());
    }
  };

  void *driver_handler = dlopen(file.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (driver_handler == nullptr) {
    std::string errmsg = file + " : dlopen failed, ";
    auto *dl_errmsg = dlerror();
    if (dl_errmsg != nullptr) {
      errmsg += dl_errmsg;
    } else {
      errmsg += "no error msg.";
    }

    MBLOG_WARN << errmsg;
    return {STATUS_INVALID, errmsg};
  }

  driver_func = (DriverDescription)dlsym(driver_handler, "DriverDescription");
  if (driver_func == nullptr) {
    std::string errmsg = file + " : dlsym DriverDescription failed, ";
    auto *dl_errmsg = dlerror();
    if (dl_errmsg != nullptr) {
      errmsg += dl_errmsg;
    } else {
      errmsg += "no error msg.";
    }

    dlclose(driver_handler);
    MBLOG_WARN << errmsg;
    return {STATUS_NOTSUPPORT, errmsg};
  }

  std::shared_ptr<Driver> driver = std::make_shared<Driver>();
  std::shared_ptr<DriverDesc> desc = driver->GetDriverDesc();
  driver_func(desc.get());
  if (DriversContains(drivers_list_, driver) == true) {
    MBLOG_DEBUG
        << "add driver: " << file
        << " failed, it already has the same function library in libraries.";
    dlclose(driver_handler);
    return {STATUS_EXIST, file + " : driver is already registered."};
  }
  desc->SetFilePath(file);
  auto no_delete = desc->GetNoDelete();
  if (no_delete) {
    auto *driver_handler_sec =
        dlopen(file.c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
    if (driver_handler_sec != nullptr) {
      dlclose(driver_handler_sec);
    } else {
      MBLOG_WARN << "dlopen " << file << " as no delete failed.";
    }
  }

  drivers_list_.push_back(driver);
  dlclose(driver_handler);

  MBLOG_DEBUG << "add driver:";
  MBLOG_DEBUG << "  name: " << desc->GetName();
  MBLOG_DEBUG << "  class: " << desc->GetClass();
  MBLOG_DEBUG << "  type: " << desc->GetType();
  MBLOG_DEBUG << "  description: " << desc->GetDescription();
  MBLOG_DEBUG << "  version: " << desc->GetVersion();
  MBLOG_DEBUG << "  driver file: " << file;
  return STATUS_OK;
}

std::vector<std::shared_ptr<Driver>> Drivers::GetAllDriverList() {
  return drivers_list_;
}

std::vector<std::shared_ptr<Driver>> Drivers::GetDriverListByClass(
    const std::string &driver_class) {
  std::vector<std::shared_ptr<Driver>> drivers_class_list;
  for (const auto &driver : drivers_list_) {
    std::shared_ptr<DriverDesc> desc_temp = driver->GetDriverDesc();
    if (desc_temp->GetClass() == driver_class) {
      drivers_class_list.push_back(driver);
    }
  }

  return drivers_class_list;
}

std::vector<std::string> Drivers::GetDriverClassList() {
  std::vector<std::string> driver_class_list;
  for (auto &driver : drivers_list_) {
    std::shared_ptr<DriverDesc> desc = driver->GetDriverDesc();
    driver_class_list.push_back(desc->GetClass());
  }

  RemoveSameElements(&driver_class_list);
  return driver_class_list;
}

std::vector<std::string> Drivers::GetDriverTypeList(
    const std::string &driver_class) {
  std::vector<std::string> driver_type_list;
  for (auto &driver : drivers_list_) {
    std::shared_ptr<DriverDesc> desc = driver->GetDriverDesc();
    if (desc->GetClass() == driver_class) {
      driver_type_list.push_back(desc->GetType());
    }
  }

  RemoveSameElements(&driver_type_list);
  return driver_type_list;
}

std::vector<std::string> Drivers::GetDriverNameList(
    const std::string &driver_class, const std::string &driver_type) {
  std::vector<std::string> driver_name_list;
  for (auto &driver : drivers_list_) {
    std::shared_ptr<DriverDesc> desc = driver->GetDriverDesc();
    if (desc->GetClass() == driver_class && desc->GetType() == driver_type) {
      driver_name_list.push_back(desc->GetName());
    }
  }

  RemoveSameElements(&driver_name_list);
  return driver_name_list;
}

std::shared_ptr<Driver> Drivers::GetDriver(const std::string &driver_class,
                                           const std::string &driver_type,
                                           const std::string &driver_name,
                                           const std::string &driver_version) {
  std::vector<std::string> driver_version_list;
  std::shared_ptr<Driver> temp_driver = nullptr;
  for (auto driver : drivers_list_) {
    std::shared_ptr<DriverDesc> desc = driver->GetDriverDesc();
    if (desc->GetClass() != driver_class || desc->GetType() != driver_type ||
        desc->GetName() != driver_name) {
      continue;
    }

    if (desc->GetVersion() == driver_version) {
      return driver;
    }

    if (temp_driver == nullptr) {
      temp_driver = driver;
      continue;
    }

    if (temp_driver->GetDriverDesc()->GetVersion() <
        driver->GetDriverDesc()->GetVersion()) {
      temp_driver = driver;
      continue;
    }
  }

  return temp_driver;
}

void Drivers::RemoveSameElements(std::vector<std::string> *driver_list) {
  sort(driver_list->begin(), driver_list->end());
  driver_list->erase(unique(driver_list->begin(), driver_list->end()),
                     driver_list->end());
}

bool Drivers::DriversContains(
    const std::vector<std::shared_ptr<Driver>> &drivers_list,
    const std::shared_ptr<Driver> &driver) {
  std::shared_ptr<DriverDesc> target_desc = driver->GetDriverDesc();
  for (const auto &driver_item : drivers_list) {
    std::shared_ptr<DriverDesc> desc = driver_item->GetDriverDesc();

    if (desc->GetClass() != target_desc->GetClass()) {
      continue;
    }

    if (desc->GetType() != target_desc->GetType()) {
      continue;
    }

    if (desc->GetName() != target_desc->GetName()) {
      continue;
    }

    if (desc->GetDescription() != target_desc->GetDescription()) {
      continue;
    }

    if (desc->GetVersion() != target_desc->GetVersion()) {
      continue;
    }

    return true;
  }

  return false;
}

void Drivers::SetDefaultScanPath(const std::string &path) {
  default_scan_path_ = path;
}

void Drivers::SetDefaultInfoPath(const std::string &path) {
  default_driver_info_path_ = path;
}

}  // namespace modelbox