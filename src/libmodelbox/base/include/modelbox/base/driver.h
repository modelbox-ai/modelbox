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

#ifndef MODELBOX_DRIVER_H_
#define MODELBOX_DRIVER_H_

#include <modelbox/base/configuration.h>
#include <modelbox/base/status.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
namespace modelbox {

constexpr const char *DRIVER_CLASS_VIRTUAL = "DRIVER-VIRTUAL";
constexpr const char *DRIVER_CLASS_INFERENCE = "DRIVER-INFERENCE";
constexpr const char *DRIVER_TYPE_VIRTUAL = "virtual";
#ifdef ANDROID
constexpr const char *DEFAULT_SCAN_INFO = "./tmp/modelbox-driver-info";
#else
constexpr const char *DEFAULT_SCAN_INFO = "/tmp/modelbox-driver-info";
#endif
constexpr const char *DEFAULT_LD_CACHE = "/etc/ld.so.cache";

class Driver;
class DriverFactory {
 public:
  DriverFactory(){};
  virtual ~DriverFactory(){};

  virtual std::shared_ptr<Driver> GetDriver() {
    return std::make_shared<Driver>();
  };

  virtual void SetDriver(std::shared_ptr<Driver> driver) { return; }

 private:
  friend class Driver;
};

class DriverDesc {
 public:
  DriverDesc() : driver_no_delete_(false), global_(false), deep_bind_(false){};
  virtual ~DriverDesc(){};
  const std::string GetClass();
  const std::string GetType();
  const std::string GetName();
  const std::string GetDescription();
  const std::string GetVersion();
  const std::string GetFilePath();
  const bool GetNoDelete();
  const bool GetGlobal();
  const bool GetDeepBind();

  void SetClass(const std::string &classname);
  void SetType(const std::string &type);
  void SetName(const std::string &name);
  void SetDescription(const std::string &description);
  Status SetVersion(const std::string &version);
  void SetFilePath(const std::string &file_path);
  void SetNodelete(const bool &no_delete);
  void SetGlobal(const bool &global);
  void SetDeepBind(const bool &deep_bind);

 protected:
  bool driver_no_delete_;
  bool global_;
  bool deep_bind_;
  std::string driver_class_;
  std::string driver_type_;
  std::string driver_name_;
  std::string driver_description_;
  std::string driver_version_;
  std::string driver_file_path_;

 private:
  Status CheckVersion(const std::string &version);
};

class DriverHandlerInfo {
 public:
  DriverHandlerInfo() : initialize_count_(0), handler_count_(0){};
  virtual ~DriverHandlerInfo(){};

  int IncHanderRefcnt() { return ++handler_count_; }
  int DecHanderRefcnt() { return --handler_count_; }

  int initialize_count_;
  int handler_count_;
  std::mutex initialize_lock_;
};

class DriverHandler {
 public:
  std::shared_ptr<DriverHandlerInfo> Add(void *driver_handler);
  modelbox::Status Remove(void *driver_handler);
  std::shared_ptr<DriverHandlerInfo> Get(void *driver_handler);

  std::mutex handler_map_lock;

 private:
  std::map<void *, std::shared_ptr<DriverHandlerInfo>> handler_map;
};

class Driver {
 public:
  Driver();
  virtual ~Driver();

  std::string GetDriverFile();

  virtual std::shared_ptr<DriverFactory> CreateFactory();

  std::shared_ptr<DriverDesc> GetDriverDesc();

  void SetDriverDesc(std::shared_ptr<DriverDesc> desc);
  bool IsVirtual();
  void SetVirtual(bool is_virtual);

 protected:
  std::shared_ptr<DriverDesc> desc_ = std::make_shared<DriverDesc>();

 private:
  int GetMode(bool no_delete, bool global, bool deep_mind);
  void CloseFactory();
  bool is_virtual_ = false;
  void *driver_handler_{nullptr};
  int factory_count_ = 0;
  std::mutex mutex_;
  std::shared_ptr<DriverFactory> factory_;
};

class VirtualDriverDesc : public DriverDesc {
 public:
  VirtualDriverDesc(){};
  virtual ~VirtualDriverDesc(){};
};

class VirtualDriver : public Driver {
 public:
  std::shared_ptr<VirtualDriverDesc> GetVirtualDriverDesc();
  void SetVirtualDriverDesc(std::shared_ptr<VirtualDriverDesc> desc);
  virtual std::shared_ptr<DriverFactory> CreateFactory();
  std::vector<std::shared_ptr<modelbox::Driver>> GetBindDriver() {
    return std::vector<std::shared_ptr<modelbox::Driver>>();
  }

 private:
  std::shared_ptr<VirtualDriverDesc> virtual_driver_desc_;
};

class Drivers;
class VirtualDriverManager : public DriverFactory {
 public:
  VirtualDriverManager();
  virtual ~VirtualDriverManager();
  virtual Status Add(const std::string &path);
  virtual Status Init(Drivers &driver);
  virtual Status Scan(std::vector<std::string> scan_dirs);
  virtual Status Scan(const std::string &path);
  std::vector<std::shared_ptr<VirtualDriver>> GetAllDriverList();
  void Clear();

 protected:
  std::vector<std::shared_ptr<VirtualDriver>> drivers_list_;
};

class DriversScanResultInfo {
 public:
  DriversScanResultInfo() = default;
  ~DriversScanResultInfo() {
    load_success_info_.clear();
    load_failed_info_.clear();
  }
  std::list<std::string> &GetLoadSuccessInfo() { return load_success_info_; }
  std::map<std::string, std::string> &GetLoadFailedInfo() {
    return load_failed_info_;
  }

 private:
  std::list<std::string> load_success_info_;
  std::map<std::string, std::string> load_failed_info_;
};
class Drivers {
 public:
  Drivers()
      : drivers_scan_result_info_(std::make_shared<DriversScanResultInfo>()){};
  virtual ~Drivers(){};

  Status Initialize(std::shared_ptr<Configuration> config);
  Status Scan();
  void Clear();
  Status Scan(const std::string &path, const std::string &filter);
  Status VirtualDriverScan();
  Status Add(const std::string &file);

  std::vector<std::string> GetDriverClassList();
  std::vector<std::string> GetDriverTypeList(const std::string &driver_class);
  std::vector<std::string> GetDriverNameList(const std::string &driver_class,
                                             const std::string &driver_type);
  std::vector<std::shared_ptr<Driver>> GetAllDriverList();
  std::vector<std::shared_ptr<Driver>> GetDriverListByClass(
      const std::string &driver_class);
  std::shared_ptr<Driver> GetDriver(const std::string &driver_class,
                                    const std::string &driver_type,
                                    const std::string &driver_name,
                                    const std::string &driver_version = "");
  static std::shared_ptr<Drivers> GetInstance();

 private:
  Status InnerScan();
  Status WriteScanInfo(const std::string &scan_info_path,
                       const std::string &check_code);
  Status GatherScanInfo(const std::string &scan_path);
  void FillCheckInfo(std::string &file_check_node,
                     std::unordered_map<std::string, bool> &file_map,
                     int64_t &ld_cache_time);
  bool CheckPathAndMagicCode();
  void PrintScanResults(const std::string &scan_path);
  void PrintScanResult(
      const std::list<std::string> &load_success_info,
      const std::map<std::string, std::string> &load_failed_info);
  void RemoveSameElements(std::vector<std::string> *driver_list);
  bool DriversContains(const std::vector<std::shared_ptr<Driver>> &drivers_list,
                       std::shared_ptr<Driver> driver);
  std::shared_ptr<Configuration> config_;
  std::vector<std::shared_ptr<Driver>> drivers_list_;
  std::vector<std::shared_ptr<VirtualDriverManager>>
      virtual_driver_manager_list_;
  std::vector<std::string> driver_dirs_;
  std::shared_ptr<DriversScanResultInfo> drivers_scan_result_info_;
  uint64_t last_modify_time_sum_{0};
};

}  // namespace modelbox

#endif  // MODELBOX_DRIVER_H_