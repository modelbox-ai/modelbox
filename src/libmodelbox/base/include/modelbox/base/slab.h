/**
 * Copyright (C) 2020 Huawei Technologies Co., Ltd. All rights reserved.
 */

#ifndef MODELBOX_SLAB_H_
#define MODELBOX_SLAB_H_

#include <modelbox/base/list.h>
#include <modelbox/base/status.h>
#include <modelbox/base/timer.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace modelbox {

/// Slab object
struct SlabObject {
  /// list head
  ListHead list;

  /// slab index in slab cache
  unsigned long index;
};

class SlabCache;

/// Memeory allocator interface
class MemoryAllocFree {
 public:
  /**
   * @brief Malloc memory
   * @param size memory size
   * @return memory pointer
   */
  virtual void *MemAlloc(size_t size) = 0;

  /**
   * @brief Free memory
   * @param ptr pointer to memory
   */
  virtual void MemFree(void *ptr) = 0;
};

/**
 * @brief Slab object
 */
class Slab {
 public:
  /**
   * @brief Construct slab
   * @param cache slab cache.
   * @param obj_size object size.
   * @param mem_size memory size.
   */
  Slab(SlabCache *cache, size_t obj_size, size_t mem_size);

  virtual ~Slab();

  /**
   * @brief Init slab, malloc memory.
   * @return success, return true, else return false.
   */
  bool Init();

  /**
   * @brief Alloc a object from slab.
   * @return allocated object pointer.
   */
  void *Alloc();

  /**
   * @brief Free a object.
   * @param ptr pointer to object
   */
  void Free(void *ptr);

  /**
   * @brief Check if slab is full
   * @return true, slab is full
   */
  bool IsFull();

  /**
   * @brief Check if address is in slab
   * @return true, address is in slab
   */
  bool IsInSlab(const void *ptr);

  /**
   * @brief Check if slab is empty.
   * @return true, slab is empty.
   */
  bool IsEmpty();

  /**
   * @brief Get object size.
   * @return int, object size
   */
  size_t ObjectSize();

  /**
   * @brief Get active object number.
   * @return int, active object number.
   */
  int ActiveObjects();

  /**
   * @brief Get total object number.
   * @return int, total object number.
   */
  int ObjectNumber();

  /**
   * @brief Get alive time.
   * @return alive time.
   */
  time_t AliveTime();

 protected:
  /**
   * @brief Alloc memory override
   * @param size memory size.
   * @return memory pointer
   */
  void *_Alloc(size_t size);

  /**
   * @brief Free memory override
   * @param ptr memory pointer
   * @return alive time.
   */
  void _Free(void *ptr);

 private:
  friend class SlabCache;

  ListHead list;
  struct SlabObject *objs_{nullptr};
  ListHead free_obj_head_;

  size_t obj_size_;
  size_t obj_num_;
  size_t active_obj_num_;

  void *mem_{nullptr};
  size_t mem_size_{0};

  SlabCache *cache_{nullptr};

  time_t last_alive_;
};

/**
 * @brief Slab cache
 */
class SlabCache {
 public:
  /**
   * @brief Construct slabcache
   * @param obj_size object size.
   * @param slab_size each slab memory size.
   * @param mem_allocator memory allocator.
   */
  SlabCache(size_t obj_size, size_t slab_size,
            MemoryAllocFree *mem_allocator = nullptr);

  virtual ~SlabCache();

  /**
   * @brief Alloc a object from slab.
   * @return shared pointer to object.
   */
  std::shared_ptr<void> AllocSharedPtr();

  /**
   * @brief Shrink slab
   * @param keep number to keep.
   * @param before shrink cache before specific time.
   */
  void Shrink(int keep = 0, time_t before = 0);

  /**
   * @brief Reclaim slab
   * @param before shrink cache before specific time.
   */
  void Reclaim(time_t before = 30);

  /**
   * @brief Get empty slab number.
   * @return empty slab number.
   */
  int GetEmptySlabNumber();

  /**
   * @brief Get total slab number.
   * @return total slab number.
   */
  uint32_t SlabNumber();

  /**
   * @brief Get object size.
   * @return object size.
   */
  size_t ObjectSize();

  /**
   * @brief Get object number.
   * @return object number.
   */
  uint32_t GetObjNumber();

  /**
   * @brief Get free object number.
   * @return free object number.
   */
  uint32_t GetFreeObjNumber();

  /**
   * @brief Get active object number.
   * @return active object number.
   */
  uint32_t GetActiveObjNumber();

 protected:
  /**
   * @brief Remove slabs
   * @param head slabe head
   */
  void RemoveSlabs(ListHead *head);

  /**
   * @brief Remove slab
   * @param s pointer to slab
   */
  void RemoveSlabLocked(Slab *s);

  /**
   * @brief Remove slab with specific args
   * @param head slabe head
   * @param count remove number
   * @param time_before time before
   */
  void RemoveSlabs(ListHead *head, size_t count, time_t time_before);

  /**
   * @brief Grow slab
   * @return grow result
   */
  bool GrowLocked(std::unique_lock<std::mutex> *lock);

  /**
   * @brief Alloc memory override
   * @param size memory size.
   * @return memory pointer
   */
  void *_Alloc(size_t size);

  /**
   * @brief Free memory override
   * @param ptr memory pointer
   * @return alive time.
   */
  void _Free(void *ptr);

 private:
  friend class Slab;

  /**
   * @brief Alloc a object from slab
   * @param obj object allocated
   * @param slab which slab
   */
  void AllocObject(void **obj, Slab **slab);

  /**
   * @brief Free a object into slab
   * @param obj object allocated
   * @param slab which slab
   */
  void FreeObject(void *obj, Slab *slab);

  size_t obj_size_{0};
  size_t slab_size_{0};

  int obj_num_{0};
  int batch_object_num_{0};
  std::atomic<uint32_t> active_obj_num_;
  std::atomic<uint32_t> slab_empty_num_;
  std::atomic<uint32_t> slab_num_;

  std::mutex lock_;

  ListHead full_;
  ListHead partial_;
  ListHead empty_;

  MemoryAllocFree *mem_allocator_{nullptr};
};

class SlabCacheReclaimer {
 public:
  static SlabCacheReclaimer &Instance();
  virtual ~SlabCacheReclaimer();

  void AddSlabCache(SlabCache *slabcache);

  void RmvSlabCache(SlabCache *slabcache);

 private:
  SlabCacheReclaimer();
  void DoReclaim();

  void StartReclaimWorker();
  void StopReclaimWoker();

  std::unordered_map<SlabCache *, SlabCache *> slab_cache_list_;
  std::mutex cache_lock_;
  Timer timer_;
  std::shared_ptr<TimerTask> reclaimer_timer_;
  std::atomic<uint32_t> slab_cache_num_;
};

}  // namespace modelbox
#endif