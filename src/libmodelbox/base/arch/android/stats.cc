#include "stats.h"

#include <memory>

#include "modelbox/base/os.h"

namespace modelbox {

AndroidOSProcess::AndroidOSProcess() {}

AndroidOSProcess::~AndroidOSProcess() {}

int32_t AndroidOSProcess::GetThreadsNumber(uint32_t pid) { return 0; }
uint32_t AndroidOSProcess::GetMemorySize(uint32_t pid) { return 0; }
uint32_t AndroidOSProcess::GetMemoryRSS(uint32_t pid) { return 0; }
uint32_t AndroidOSProcess::GetMemorySHR(uint32_t pid) { return 0; }
uint32_t AndroidOSProcess::GetPid() { return 0; }

std::vector<uint32_t> AndroidOSProcess::GetProcessTime(uint32_t pid) {
  return {};
};

std::vector<uint32_t> AndroidOSProcess::GetTotalTime(uint32_t pid) {
  return {};
};

}  // namespace modelbox