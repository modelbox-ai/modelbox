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

#ifndef MODELBOX_MEMORY_POOL_H
#define MODELBOX_MEMORY_POOL_H

#include <modelbox/base/collector.h>
#include <modelbox/base/slab.h>

namespace modelbox {

/**
 * @brief Memory pool interface
 */
class MemoryPoolBase : public MemoryAllocFree,
                       public std::enable_shared_from_this<MemoryPoolBase> {
 public:
  /**
   * @brief Initialize slab cache.
   * @param low low bit 2^low
   * @param high high bit 2^high
   * @return result.
   */
  Status InitSlabCache(int low = 5, int high = 27);

  /**
   * @brief Alloc a object from slab.
   * @return shared pointer to object.
   */
  std::shared_ptr<void> AllocSharedPtr(size_t size);

  /**
   * @brief Shrink slab cache.
   * @param each_keep slab number for each to keep.
   * @param before shrink slab before specific time.
   * @param expire force shrink cache before specific time.
   * @return shrink result.
   */
  virtual Status ShrinkSlabCache(int each_keep, time_t before,
                                 time_t expire = 0);

  /**
   * @brief Get all slab object number
   * @return return total object number
   */
  uint32_t GetAllObjectNum();

  /**
   * @brief Get all active slab object number
   * @return return total object number
   */
  uint32_t GetAllActiveObjectNum();

  /**
   * @brief Destroy slab cache.
   */
  void DestroySlabCache();

  /**
   * @brief Get the vector of slab cache pointers
   * @return the vector of slab cach pointers
   */
  std::vector<std::shared_ptr<SlabCache>> GetSlabCaches();

  /**
   * @brief Set memory pool name
   * @param name
   */
  void SetName(std::string name);

  /**
   * @brief Get memory pool name
   *
   * @return std::string
   */
  std::string GetName();

  static std::vector<std::shared_ptr<MemoryPoolBase>> GetAllPools();

  MemoryPoolBase();
  MemoryPoolBase(std::string name);
  virtual ~MemoryPoolBase();

 protected:
  /**
   * @brief Create a slab cache.
   * @param obj_size object size.
   * @param slab_size slab size.
   * @return shared pointer to slabcache.
   */
  virtual std::shared_ptr<SlabCache> MakeSlabCache(size_t obj_size,
                                                   size_t slab_size);

  /**
   * @brief Calculate slabcache size.
   * @param object_size object size.
   * @return slab size.
   */
  virtual size_t CalSlabSize(size_t object_size);

  /**
   * @brief Alloc Memory
   * @param size memory size.
   * @return pointer to memory.
   */
  void *MemAlloc(size_t size) override;

  /**
   * @brief Free Memory
   * @param ptr to memory.
   */
  void MemFree(void *ptr) override;

  /**
   * @brief Add a new slab cache
   * @param slab_cache new slab cache.
   */
  void AddSlabCache(const std::shared_ptr<SlabCache> &slab_cache);

  /**
   * @brief Clear all slabs 
   */
  void ClearAllSlabs();
  
 private:
  std::vector<std::shared_ptr<SlabCache>> slab_caches_;
  std::string pool_name_;
  static std::map<MemoryPoolBase *, std::weak_ptr<MemoryPoolBase>> pool_list_;
  static std::mutex pool_list_lock_;
};

}  // namespace modelbox

#endif  // MODELBOX_MEMORY_POOL_H
