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

#include "modelbox/base/memory_pool.h"

#include <memory>

#include "modelbox/base/collector.h"
#include "modelbox/base/log.h"

namespace modelbox {

const size_t kMaxSlabCacheSize = 128 * 1024 * 1024;

void *MemoryPoolBase::MemAlloc(size_t size) { return malloc(size); }

void MemoryPoolBase::MemFree(void *ptr) { free(ptr); }

std::shared_ptr<void> MemoryPoolBase::AllocSharedPtr(size_t size) {
  std::shared_ptr<void> ret = nullptr;

  if (size <= 0) {
    return nullptr;
  }

  auto alloc_sharedptr = [&]() -> std::shared_ptr<void> {
    for (auto &cache : slab_caches_) {
      if (size > cache->ObjectSize()) {
        continue;
      }

      ret = cache->AllocSharedPtr();
      if (ret == nullptr) {
        continue;
      }

      break;
    }

    if (ret == nullptr) {
      auto *ptr = MemAlloc(size);
      if (ptr == nullptr) {
        return nullptr;
      }
      ret.reset(ptr, [this](void *ptr) { this->MemFree(ptr); });
    }

    return ret;
  };

  ret = alloc_sharedptr();
  if (ret == nullptr) {
    ShrinkSlabCache(0, 0, 0);
    ret = alloc_sharedptr();
  }

  return ret;
}

Status MemoryPoolBase::ShrinkSlabCache(int each_keep, time_t before,
                                       time_t expire) {
  for (auto &cache : slab_caches_) {
    cache->Shrink(each_keep, before);
    if (expire > before) {
      cache->Shrink(0, expire);
    }
  }

  return STATUS_OK;
}

uint32_t MemoryPoolBase::GetAllObjectNum() {
  uint32_t total_number = 0;
  for (auto &cache : slab_caches_) {
    total_number += cache->GetObjNumber();
  }

  return total_number;
}

uint32_t MemoryPoolBase::GetAllActiveObjectNum() {
  uint32_t total_number = 0;
  for (auto &cache : slab_caches_) {
    total_number += cache->GetActiveObjNumber();
  }

  return total_number;
}

std::vector<std::shared_ptr<SlabCache>> MemoryPoolBase::GetSlabCaches() {
  return slab_caches_;
}

void MemoryPoolBase::DestroySlabCache() { slab_caches_.clear(); }

void MemoryPoolBase::RegisterCollector(const std::string &name) {
  GetInstance()->AddObject(name, shared_from_this());
}

void MemoryPoolBase::UnregisterCollector(const std::string &name) {
  GetInstance()->RmvObject(name);
}

Collector<MemoryPoolBase> *MemoryPoolBase::GetInstance() {
  static Collector<MemoryPoolBase> instance;
  return &instance;
}

std::shared_ptr<SlabCache> MemoryPoolBase::MakeSlabCache(size_t obj_size,
                                                         size_t slab_size) {
  return std::make_shared<SlabCache>(obj_size, slab_size, this);
}

void MemoryPoolBase::AddSlabCache(std::shared_ptr<SlabCache> slab_cache) {
  slab_caches_.push_back(slab_cache);
  std::sort(slab_caches_.begin(), slab_caches_.end(),
            [](std::shared_ptr<SlabCache> a, std::shared_ptr<SlabCache> b) {
              return a->ObjectSize() < b->ObjectSize();
            });
}

size_t MemoryPoolBase::CalSlabSize(size_t object_size) {
  const size_t size_1K = 1024;
  const size_t size_1M = 1024 * 1024;

  if (object_size <= size_1K) {
    return size_1M;
  } else if (object_size <= 16 * size_1K) {
    return 8 * size_1M;
  } else if (object_size <= 512 * size_1K) {
    return 16 * size_1M;
  } else if (object_size <= size_1M) {
    return 8 * size_1M;
  } else if (object_size <= 2 * size_1M) {
    return 16 * size_1M;
  } else if (object_size <= 4 * size_1M) {
    return 32 * size_1M;
  } else if (object_size <= 8 * size_1M) {
    return 32 * size_1M;
  } else if (object_size <= 16 * size_1M) {
    return 64 * size_1M;
  } else if (object_size <= 32 * size_1M) {
    return 64 * size_1M;
  } else if (object_size <= 64 * size_1M) {
    return 128 * size_1M;
  } else if (object_size <= 128 * size_1M) {
    return 128 * size_1M;
  }

  return 0;
}

Status MemoryPoolBase::InitSlabCache(int low, int high) {
  std::shared_ptr<SlabCache> slab;
  const unsigned long shift_low = low;
  const unsigned long shift_high = high;
  for (unsigned long i = shift_low; i <= shift_high; i++) {
    size_t obj_size = 1 << i;
    if (obj_size > kMaxSlabCacheSize) {
      MBLOG_WARN << "Unsupport cache size, max is "
                 << GetBytesReadable(kMaxSlabCacheSize);
      break;
    }
    size_t slab_size;
    slab_size = CalSlabSize(obj_size);
    if (slab_size == 0) {
      break;
    }
    slab = MakeSlabCache(obj_size, slab_size);
    AddSlabCache(slab);
  }

  return STATUS_OK;
}

}  // namespace modelbox