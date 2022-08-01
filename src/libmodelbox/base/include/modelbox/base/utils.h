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

#ifndef MODELBOX_UTILS_H_
#define MODELBOX_UTILS_H_

#include <modelbox/base/status.h>
#include <openssl/ssl.h>
#include <sys/time.h>
#include <time.h>

#include <functional>
#include <list>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

namespace modelbox {

#define UNUSED_VAR(var) \
  { auto &unused __attribute__((unused)) = var; }

#define MODELBOX_DLL_PUBLIC __attribute__((visibility("default")))
#define MODELBOX_DLL_LOCAL __attribute__((visibility("hidden")))

/**
 * @brief The defer statement pushes a function call onto the guard; the
 * saved calls in called when the function returns
 */
class DeferGuard {
 public:
  template <class Callable>

  /**
   * @brief Defer guard with function call
   * @param fn function
   */
  // NOLINTNEXTLINE
  DeferGuard(Callable &&fn) : fn_(std::forward<Callable>(fn)) {}

  /**
   * @brief Copy constructor
   * @param other another defer guard.
   */
  DeferGuard(DeferGuard &&other);

  virtual ~DeferGuard();

  DeferGuard(const DeferGuard &) = delete;
  void operator=(const DeferGuard &) = delete;

 private:
  std::function<void()> fn_;
};

/**
 * @brief The defer statement pushes a function call onto a list; the list of
 * saved calls in called when the function returns and condition return true
 */
class DeferGuardChain {
 public:
  /**
   * @brief Defer guard with function call
   * @param fn function
   */
  template <class Callable>
  // NOLINTNEXTLINE
  DeferGuardChain(Callable &&fn) : fn_cond_(std::forward<Callable>(fn)) {}

  /**
   * @brief Defer guard with function call
   * @param other other guard
   */
  DeferGuardChain(DeferGuardChain &&other);

  /**
   * @brief Add function to list
   * @param fn function
   */
  DeferGuardChain &operator+=(std::function<void()> &&fn);

  virtual ~DeferGuardChain();

  DeferGuardChain(const DeferGuardChain &) = delete;
  void operator=(const DeferGuardChain &) = delete;

 private:
  std::list<std::function<void()>> fn_list_;
  std::function<bool()> fn_cond_;
};

/**
 * @brief The function list will be called when function returns
 */
#define DeferCond ::modelbox::DeferGuardChain __defer_cond = [&]()

/**
 * @brief Add defer function to list.
 */
#define DeferCondAdd __defer_cond += [&]()

#define MODELBOX_CONCAT_(a, b) a##b
#define MODELBOX_CONCAT(a, b) MODELBOX_CONCAT_(a, b)

/**
 * @brief Call when the function returns
 */
#define Defer \
  ::modelbox::DeferGuard MODELBOX_CONCAT(__defer__, __LINE__) = [&]()

/**
 * @brief Extend defer with capture list args
 */
#define DeferExt(...)                               \
  ::modelbox::DeferGuard MODELBOX_CONCAT(__defer__, \
                                         __LINE__) = [##__VA_ARGS__]()

enum LIST_FILE_TYPE : unsigned int {
  LIST_FILES_ALL = 0x3,
  LIST_FILES_FILE = 0x1,
  LIST_FILES_DIR = 0x2,
};

/**
 * @brief List files or directoires in path of directory
 * @param path path to directory
 * @param filter list filter
 * @param listfiles files or dirs result
 * @param type list type.
 * @return list result
 */
Status ListFiles(const std::string &path, const std::string &filter,
                 std::vector<std::string> *listfiles,
                 enum LIST_FILE_TYPE type = LIST_FILES_ALL);

/**
 * @brief find the earilest created file index in path
 * @param listfiles the vector of files
 * @return the earilest file index
 */
size_t FindTheEarliestFileIndex(std::vector<std::string> &listfiles);

/**
 * @brief List files in path of directory and sub directories.
 * @param path path to directory
 * @param filter list filter
 * @param listfiles files or dirs result
 * @return list result
 */
Status ListSubDirectoryFiles(const std::string &path, const std::string &filter,
                             std::vector<std::string> *listfiles);

/**
 * @brief Create directory recursively
 * @param path path to directory
 * @return create result
 */
Status CreateDirectory(const std::string &path);

/**
 * @brief Revmoe directory recursively
 *
 * @param path path to directory
 * @return Status remove result
 */
void RemoveDirectory(const std::string &path);

/**
 * @brief judge if the path is directory
 *
 * @param path path to be judged
 * @return true means directory
 */
bool IsDirectory(const std::string &path);

/**
 * @brief Copy from from src to dest
 * @param src copy file from
 * @param dest copy file to
 * @param mode dest file mode
 * @param overwrite whether overwrite existing file
 * @return Copy result
 */
Status CopyFile(std::string src, std::string dest, int mode = 0,
                bool overwrite = false);

/**
 * @brief Get current time, in usecond
 * @return Current time in usecond
 */
int64_t GetCurrentTime();

/**
 * @brief Check whether path is absolute
 * @param path path to check
 * @return is absolute
 */
bool IsAbsolutePath(const std::string &path);

/**
 * @brief Get directory name of path
 * @param path path
 * @return directory name
 */
std::string GetDirName(const std::string &path);

/**
 * @brief Get basename
 * @param path path
 * @return basename, empty when fail
 */
std::string GetBaseName(const std::string &path);

/**
 * @brief Get random number
 * @param buf output number
 * @param num length of number
 */
void GetRandom(unsigned char *buf, int num);

/**
 * @brief Canonicalize path
 * @param path path
 * @param root_path root path
 * @return path canonicalize
 */
std::string PathCanonicalize(const std::string &path,
                             const std::string &root_path = "");

inline size_t Volume(const std::vector<size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), (size_t)1,
                         std::multiplies<size_t>());
}

/**
 * @brief Calculator volume size by shapes
 * @param shapes input shapes
 * @return volume size
 */
inline size_t Volume(const std::vector<std::vector<size_t>> &shapes) {
  size_t size = 0;
  for (const auto &shape : shapes) {
    size += std::accumulate(shape.begin(), shape.end(), (size_t)1,
                            std::multiplies<size_t>());
  }

  return size;
}

/**
 * @brief Regex pattern match
 * @param str input string
 * @param pattern pattern
 * @return whether match
 */
inline bool RegexMatch(const std::string &str, const std::string &pattern) {
  std::regex re(pattern);
  return std::regex_match(str, re);
}

/**
 * @brief Split string by delim
 * @param s input string
 * @param delim delimiter
 * @return strings splitted
 */
std::vector<std::string> StringSplit(const std::string &s, char delim);

/**
 * @brief Replace string
 *
 * @param str string to replace
 * @param from replace from
 * @param to output replaced string
 */
void StringReplaceAll(std::string &str, const std::string &from,
                      const std::string &to);

/**
 * @brief Get current call stack trace
 * @param skip skip frame number
 * @param maxdepth max call stack depth
 * @return stack trace in vector list.
 */
std::vector<std::tuple<void *, std::string>> GetStacks(int skip = 0,
                                                       int maxdepth = -1);

/**
 * @brief Get current call stack trace
 * @param skip skip frame number
 * @param maxdepth max call stack depth
 * @return stack trace.
 */
std::string GetStackTrace(int skip = 0, int maxdepth = -1);

/**
 * @brief Get symbol name by addr
 * @param addr address to get symbol
 * @return symbol name, and base address.
 */
std::tuple<void *, std::string> GetSymbol(void *addr);

/**
 * @brief Convert size in integer to readable string
 * @param size size in integer
 * @return size in string
 */
std::string GetBytesReadable(size_t size);

/**
 * @brief Convert size in string to integer, format like B, Mb, GB, TB
 * @param size size in string, format like B, Mb, GB, TB
 * @return size in integer
 */
uint64_t GetBytesFromReadable(const std::string &size);

/**
 * @brief Get system tick count
 * @return current system tick count
 */
unsigned long GetTickCount();

/**
 * @brief Convert json to toml
 * @param json_data json data
 * @param toml_data toml data converted
 * @return Convert result
 */
Status JsonToToml(const std::string &json_data, std::string *toml_data);

/**
 * @brief Convert toml to json
 * @param toml_data toml data
 * @param json_data json data converted
 * @param readable wheather output with format
 * @return Convert result
 */
Status TomlToJson(const std::string &toml_data, std::string *json_data,
                  bool readable = false);

/**
 * @brief hardening SSL
 * @param ctx ssl context
 * @return hardening result
 */
Status HardeningSSL(SSL_CTX *ctx);

/**
 * @brief Get errno in string
 * @param errnum error number
 * @return errno in string
 */

std::string StrError(int errnum);

/**
 * @brief Get compiled time
 * @return version
 */
const char *GetModelBoxVersion();

}  // namespace modelbox
#endif  // MODELBOX_UTILS_H_