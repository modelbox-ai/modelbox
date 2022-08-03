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

#include "modelbox/base/utils.h"

#include <cxxabi.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <ftw.h>
#include <glob.h>
#include <libgen.h>
#include <modelbox/base/log.h>
#include <openssl/rand.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "modelbox/base/config.h"
#include "securec.h"

namespace modelbox {

constexpr int MAX_STACK_DEPTH = 100;
#ifdef _WIN32
const char Separator = '\\';
#else
const char Separator = '/';
#endif

DeferGuard::DeferGuard(DeferGuard &&other) noexcept : fn_(std::move(other.fn_)) {
  other.fn_ = nullptr;
}

DeferGuard::~DeferGuard() {
  if (fn_) {
    fn_();
  }
}

DeferGuardChain::DeferGuardChain(DeferGuardChain &&other) noexcept
    : fn_cond_(std::move(other.fn_cond_)) {
  other.fn_cond_ = nullptr;
}

DeferGuardChain &DeferGuardChain::operator+=(std::function<void()> &&fn) {
  fn_list_.push_front(std::move(fn));
  return *this;
}

DeferGuardChain::~DeferGuardChain() {
  if (!fn_cond_()) {
    return;
  }

  for (const auto &fn : fn_list_) {
    fn();
  }
}

Status ListFiles(const std::string &path, const std::string &filter,
                 std::vector<std::string> *listfiles,
                 enum LIST_FILE_TYPE type) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == -1) {
    std::string msg = path + " does not exist, ";
    return {STATUS_NOTFOUND, msg + StrError(errno)};
  }

  if (S_ISDIR(buffer.st_mode) == 0) {
    std::string msg = path + " is not a directory, ";
    return {STATUS_INVALID, msg};
  }

  glob_t glob_result;
  std::string path_pattern = path + "/" + filter;

  auto ret = glob(path_pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
  if (ret != 0) {
    if (ret == GLOB_NOMATCH) {
      return STATUS_OK;
    }

    MBLOG_ERROR << "glob " << path_pattern << " failed, code: " << ret;
    return {STATUS_INVALID, "error code :" + std::to_string(ret)};
  }

  for (unsigned int i = 0; i < glob_result.gl_pathc; i++) {
    if (stat(glob_result.gl_pathv[i], &buffer) == -1) {
      continue;
    }

    if (S_ISDIR(buffer.st_mode) && (type & LIST_FILES_DIR)) {
      listfiles->push_back(glob_result.gl_pathv[i]);
    }

    if (S_ISDIR(buffer.st_mode) == 0 && (type & LIST_FILES_FILE)) {
      listfiles->push_back(glob_result.gl_pathv[i]);
    }
  }

  globfree(&glob_result);
  return STATUS_OK;
}

size_t FindTheEarliestFileIndex(std::vector<std::string> &listfiles) {
  struct stat buffer;
  __time_t min_sec = 0x7fffffff;
  size_t index = 0;
  for (size_t i = 0; i < listfiles.size(); ++i) {
    if (stat(listfiles[i].c_str(), &buffer) == -1) {
      MBLOG_WARN << "stat " << listfiles[i]
                 << " failed, errno: " << StrError(errno);
      continue;
    }

    if (buffer.st_mtim.tv_sec < min_sec) {
      min_sec = buffer.st_mtim.tv_sec;
      index = i;
    }
  }

  return index;
}

Status ListSubDirectoryFiles(const std::string &path, const std::string &filter,
                             std::vector<std::string> *listfiles) {
  struct stat buffer;
  DIR *pDir;
  struct dirent *ptr = nullptr;

  auto status = ListFiles(path, filter, listfiles);

  pDir = opendir(path.c_str());
  if (pDir == nullptr) {
    return {STATUS_NOTFOUND, StrError(errno)};
  }

  Defer {
    if (closedir(pDir) != 0) {
      MBLOG_WARN << "Close dir failed.";
    }
  };

  while ((ptr = readdir(pDir)) != nullptr) {
    std::string temp_path = path + "/" + std::string(ptr->d_name);
    if (stat(temp_path.c_str(), &buffer) == -1) {
      MBLOG_WARN << "stat " << temp_path
                 << " failed, errno: " << StrError(errno);
      continue;
    };

    if (S_ISDIR(buffer.st_mode) == 0) {
      continue;
    }

    if (strncmp(ptr->d_name, ".", PATH_MAX - 1) == 0 ||
        strncmp(ptr->d_name, "..", PATH_MAX - 1) == 0) {
      continue;
    }

    auto status = ListFiles(temp_path, filter, listfiles);
  }

  return STATUS_OK;
}

Status CreateDirectory(const std::string &path) {
  std::string directory_path = path + "/";
  uint32_t dir_path_len = directory_path.length();
  if (dir_path_len > PATH_MAX) {
    return STATUS_INVALID;
  }

  char dir_path[PATH_MAX] = {0};
  for (uint32_t i = 0; i < dir_path_len; ++i) {
    dir_path[i] = directory_path[i];
    if (dir_path[i] != '/') {
      continue;
    }

    if (access(dir_path, 0) == 0) {
      continue;
    }

    int32_t ret = mkdir(dir_path, 0700);
    if (ret != 0) {
      return {STATUS_FAULT, StrError(errno)};
    }
  }

  return STATUS_OK;
}

bool IsDirectory(const std::string &path) {
  struct stat buffer;
  if (stat(path.c_str(), &buffer) == -1) {
    return false;
  }

  if (S_ISDIR(buffer.st_mode) == 0) {
    return false;
  }

  return true;
}

static int rmfiles(const char *pathname, const struct stat *sbuf, int type,
                   struct FTW *ftwb) {
  remove(pathname);
  return 0;
}

void RemoveDirectory(const std::string &path) {
  nftw(path.c_str(), rmfiles, 10, FTW_DEPTH | FTW_MOUNT | FTW_PHYS);
}

Status CopyFile(const std::string &src, const std::string &dest, int mode,
                bool overwrite) {
  if (overwrite == false && access(dest.c_str(), F_OK) == 0) {
    return STATUS_FAULT;
  }

  std::ifstream src_file(src, std::ios::binary);
  std::ofstream dst_file(dest, std::ios::binary | std::ios::trunc);
  bool copy_fail = false;

  if (src_file.fail() || dst_file.fail()) {
    return STATUS_FAULT;
  }

  std::istreambuf_iterator<char> begin_source(src_file);
  std::istreambuf_iterator<char> end_source;
  std::ostreambuf_iterator<char> begin_dest(dst_file);
  std::copy(begin_source, end_source, begin_dest);

  src_file.seekg(0, std::ios::end);
  if (dst_file.tellp() != src_file.tellg()) {
    copy_fail = true;
  }

  src_file.close();
  if (dst_file.fail() || copy_fail) {
    dst_file.close();
    remove(dest.c_str());
    return false;
  }
  dst_file.close();

  if (mode) {
    chmod(dest.c_str(), mode);
  }

  return STATUS_OK;
}

int64_t GetCurrentTime() {
  static const int64_t SECOND_TO_MICRO = 1000000;
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * SECOND_TO_MICRO + tv.tv_usec;
}

std::vector<std::string> StringSplit(const std::string &s, const char delim) {
  std::vector<std::string> ret;
  std::string::size_type lastPos = s.find_first_not_of(delim, 0);
  std::string::size_type pos = s.find_first_of(delim, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    ret.push_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delim, pos);
    pos = s.find_first_of(delim, lastPos);
  }

  return ret;
}

std::vector<void *> GetStackFrames(int skip, int maxdepth) {
  std::vector<void *> stack;
  int size;

  if (maxdepth <= 0) {
    maxdepth = MAX_STACK_DEPTH;
  }

  skip++;
  stack.resize(maxdepth + skip);
  size = backtrace(&stack[0], maxdepth + skip);
  size = size - skip;

  if (size < 0) {
    stack.resize(0);
    return stack;
  }

  stack.erase(stack.begin(), stack.begin() + skip);
  stack.resize(size);

  return stack;
}

std::string DemangleCPPSymbol(const char *symbol) {
  int ret;
  std::string symbolname;
  char *name;

  name = abi::__cxa_demangle(symbol, nullptr, nullptr, &ret);
  if (ret == -2 || ret == -3 || name == nullptr) {
    symbolname = symbol;
  } else if (ret == 0) {
    symbolname = name;
  }

  if (name) {
    free(name);
  }

  return symbolname;
}

std::tuple<void *, std::string> GetSymbol(void *addr) {
  Dl_info info;
  std::ostringstream os;

  if (dladdr(addr, &info)) {
    std::string symname;
    if (info.dli_sname) {
      symname = DemangleCPPSymbol(info.dli_sname);
    }

    if (symname.length() == 0) {
      symname = "?? ()";
    }

    os << symname;
    if (info.dli_fname) {
      auto *offset = (void *)((char *)(addr) - (char *)(info.dli_fbase));
      os << " from " << info.dli_fname << "+" << offset;
    }
  } else {
    os << "?? ()";
  }

  return std::make_tuple(addr, os.str());
}

std::vector<std::tuple<void *, std::string>> GetStacks(int skip, int maxdepth) {
  std::vector<std::tuple<void *, std::string>> stacks;

  auto frames = GetStackFrames(skip + 1, maxdepth);
  stacks.reserve(frames.size());
  for (auto &frame : frames) {
    stacks.push_back(GetSymbol(frame));
  }

  return stacks;
}

std::string GetStackTrace(int skip, int maxdepth) {
  std::ostringstream os;
  const int w = sizeof(char *) * 2;
  int index = 0;
  auto frames = GetStackFrames(skip + 1, maxdepth);

  for (auto &frame : frames) {
    void *addr = nullptr;
    std::string symbol;
    std::tie(addr, symbol) = GetSymbol(frame);
    os << "#" << std::dec << index << " ";
    os << "0x" << std::setfill('0') << std::setw(w) << std::hex
       << (unsigned long)addr;
    os << ": " << symbol << std::endl;
    index++;
  }

  return os.str();
}

std::string GetBytesReadable(size_t size) {
  const char *suffix[] = {"B", "KB", "MB", "GB", "TB", "PB"};
  char length = sizeof(suffix) / sizeof(suffix[0]);

  int i = 0;
  double double_size = size;

  if (size >= 1024) {
    for (i = 0; (size / 1024) > 0 && i < length - 1; i++, size /= 1024) {
      double_size = size / 1024.0;
    }
  }

  char output[32];
  auto ret = snprintf_s(output, sizeof(output), sizeof(output) - 1, "%g%s",
                        double_size, suffix[i]);
  if (ret < 0 || ret == sizeof(output)) {
    return "";
  }

  return output;
}

uint64_t GetBytesFromReadable(const std::string &size) {
  const char *suffix[] = {"B", "K", "M", "G", "T", "P"};
  char length = sizeof(suffix) / sizeof(suffix[0]);

  uint64_t double_size = 1;
  auto ret = std::stod(size);

  auto uppercase_size = size;
  std::transform(uppercase_size.begin(), uppercase_size.end(),
                 uppercase_size.begin(), ::toupper);

  for (int i = 1; i < length; i++) {
    double_size *= 1024;
    if (uppercase_size.find(suffix[i]) == std::string::npos) {
      continue;
    }

    ret *= double_size;
  }

  return (uint64_t)ret;
}

unsigned long GetTickCount() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

bool IsAbsolutePath(const std::string &path) {
  for (const char &c : path) {
    if (c == ' ') {
      continue;
    }

    if (c != Separator) {
      return false;
    }

    return true;
  }

  return false;
}

std::string GetDirName(const std::string &path) {
  std::vector<char> path_data(path.begin(), path.end());
  path_data.push_back(0);
  auto *result = dirname(path_data.data());
  return result;
}

std::string GetBaseName(const std::string &path) {
  std::vector<char> path_data(path.begin(), path.end());
  path_data.push_back(0);
  auto *result = basename(path_data.data());
  if (result == nullptr) {
    return "";
  }
  return result;
}

void GetRandom(unsigned char *buf, int num) { RAND_bytes(buf, num); }

std::string PathCanonicalize(const std::string &path,
                             const std::string &root_path) {
  int skip_num = 0;
  size_t i = 0;
  std::string resolve_path = root_path;
  std::deque<std::string> str;
  std::vector<std::string> fields = StringSplit(path, '/');

  for (auto itr = fields.rbegin(); itr != fields.rend(); itr++) {
    if (itr->empty() || *itr == ".") {
      continue;
    }

    if (*itr == "..") {
      ++skip_num;
      continue;
    }

    if (skip_num <= 0) {
      str.push_front(*itr);
    } else {
      --skip_num;
    }
  }

  for (i = 0; i < str.size(); i++) {
    resolve_path += "/";
    resolve_path += str[i];
  }

  if (resolve_path.length() == 0) {
    return "/";
  }

  return resolve_path;
}

void StringReplaceAll(std::string &str, const std::string &from,
                      const std::string &to) {
  size_t start_pos = 0;
  if (from.empty()) {
    return;
  }

  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

Status HardeningSSL(SSL_CTX *ctx) {
  SSL_CTX_set_options(ctx, SSL_OP_NO_SSLv2);
  SSL_CTX_set_options(ctx, SSL_OP_NO_SSLv3);
  SSL_CTX_set_options(ctx, SSL_OP_NO_TLSv1);
  SSL_CTX_set_options(ctx, SSL_OP_NO_TLSv1_1);
  std::string tls1_2_safer_ciphers =
      "DHE-RSA-AES256-GCM-SHA384:"
      "ECDHE-RSA-AES256-GCM-SHA384:"
      "ECDHE-ECDSA-AES256-GCM-SHA384:"
      "DHE-RSA-AES128-GCM-SHA256:"
      "ECDHE-RSA-AES128-GCM-SHA256:"
      "ECDHE-ECDSA-AES128-GCM-SHA256:"
      "@STRENGTH";
  const auto &tls1_2_ciphers = tls1_2_safer_ciphers;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
  std::string tls1_2_suppoert_ciphers =
      "ECDHE-RSA-AES128-SHA256:"
      "ECDHE-DSS-AES128-SHA256:"
      "ECDHE-RSA-AES256-SHA256:"
      "ECDHE-DSS-AES256-SHA256:"
      "ECDHE-RSA-AES128-SHA:"
      "ECDHE-DSS-AES128-SHA:"
      "ECDHE-RSA-AES256-SHA:"
      "ECDHE-DSS-AES256-SHA:";
  tls1_2_ciphers += tls1_2_suppoert_ciphers;
#endif
  SSL_CTX_set_cipher_list(ctx, tls1_2_ciphers.data());

  return STATUS_OK;
}

std::string StrError(int errnum) {
  char buf[32];
  return strerror_r(errnum, buf, sizeof(buf));
}

void GetCompiledTime(struct tm *compiled_time) {
  char s_month[5];
  int month;
  int day;
  int year;
  int hour;
  int min;
  int sec;
  static const char *month_names = "JanFebMarAprMayJunJulAugSepOctNovDec";

  sscanf_s(__DATE__, "%4s %d %d", s_month, 4, &day, &year);
  month = (strstr(month_names, s_month) - month_names) / 3;
  sscanf_s(__TIME__, "%d:%d:%d", &hour, &min, &sec);
  compiled_time->tm_year = year - 1900;
  compiled_time->tm_mon = month;
  compiled_time->tm_mday = day;
  compiled_time->tm_isdst = -1;
  compiled_time->tm_hour = hour;
  compiled_time->tm_min = min;
  compiled_time->tm_sec = sec;
}

const char *GetModelBoxVersion() {
  static char str_ver[64] = {0};
  struct tm tm;
  GetCompiledTime(&tm);
  snprintf_s(str_ver, sizeof(str_ver), sizeof(str_ver),
             "%d.%d.%d (Build: %.4d%.2d%.2d-%.2d%.2d%.2d)",
             MODELBOX_VERSION_MAJOR, MODELBOX_VERSION_MINOR,
             MODELBOX_VERSION_PATCH, tm.tm_year + 1900, tm.tm_mon + 1,
             tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  return str_ver;
}

}  // namespace modelbox