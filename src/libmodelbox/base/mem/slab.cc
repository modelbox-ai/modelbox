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


#include "modelbox/base/slab.h"

#include <stdarg.h>
#include <stdio.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>

#include "modelbox/base/log.h"

namespace modelbox {

Slab::Slab(SlabCache *cache, size_t obj_size, size_t mem_size) {
  if (obj_size <= 0) {
    Abort("object size is invalid.");
  }

  ListInit(&free_obj_head_);
  ListInit(&list);
  mem_size_ = mem_size;
  obj_size_ = obj_size;
  cache_ = cache;

  obj_num_ = mem_size_ / obj_size;
  if (obj_num_ == 0) {
    Abort("object number is invalid.");
  }

  active_obj_num_ = 0;
  last_alive_ = time(0);
}

Slab::~Slab() {
  if (active_obj_num_ > 0) {
    MBLOG_ERROR << "active obj number: " << active_obj_num_;
    Abort("Still exist active object.");
  }

  if (objs_) {
    free(objs_);
    objs_ = nullptr;
  }

  if (mem_) {
    _Free(mem_);
    mem_ = nullptr;
  }
}

bool Slab::Init() {
  objs_ = (struct SlabObject *)malloc(sizeof(struct SlabObject) * obj_num_);
  if (objs_ == nullptr) {
    return false;
  }

  for (size_t i = 0; i < obj_num_; i++) {
    objs_[i].index = i;
    ListAddTail(&objs_[i].list, &free_obj_head_);
  }

  mem_ = _Alloc(mem_size_);
  if (mem_ == nullptr) {
    free(objs_);
    objs_ = nullptr;
    return false;
  }

  return true;
}

time_t Slab::AliveTime() {
  if (active_obj_num_ > 0) {
    return time(0);
  }

  return last_alive_;
}

void *Slab::Alloc() {
  if (ListEmpty(&free_obj_head_) || mem_ == nullptr) {
    return nullptr;
  }

  struct SlabObject *sobj =
      ListFirstEntry(&free_obj_head_, struct SlabObject, list);
  if (sobj == nullptr) {
    return nullptr;
  }

  ListDel(&sobj->list);
  active_obj_num_++;
  return (char *)mem_ + obj_size_ * sobj->index;
};

void Slab::Free(void *ptr) {
  size_t offset = (char *)ptr - (char *)mem_;
  if (offset % obj_size_ != 0) {
    Abort("Memory address is invalid, not point to object.");
  }

  size_t index = offset / obj_size_;
  if (index >= obj_num_) {
    Abort("Memory address is out of range.");
  }

  struct SlabObject *sobj = &objs_[index];
  if (!ListEntryNotInList(&sobj->list) || sobj->index != index) {
    MBLOG_ERROR << "object in list: " << ListEntryNotInList(&sobj->list);
    MBLOG_ERROR << "object index:" << sobj->index << ", free index:" << index;
    Abort("Memory is corrupted or double free.");
  }

  ListAdd(&sobj->list, &free_obj_head_);
  active_obj_num_--;

  if (active_obj_num_ == 0) {
    last_alive_ = time(0);
  }
}

bool Slab::IsFull() {
  if (ListEmpty(&free_obj_head_)) {
    return true;
  }

  return false;
}

bool Slab::IsInSlab(const void *ptr) {
  if ((char *)ptr >= (char *)mem_ && (char *)ptr <= (char *)mem_ + mem_size_) {
    return true;
  }

  return false;
}

bool Slab::IsEmpty() { return active_obj_num_ == 0; }

size_t Slab::ObjectSize() { return obj_size_; }

int Slab::ActiveObjects() { return active_obj_num_; }

int Slab::ObjectNumber() { return obj_num_; }

void *Slab::_Alloc(size_t size) {
  if (cache_) {
    return cache_->_Alloc(size);
  }
  return malloc(size);
};

void Slab::_Free(void *ptr) {
  if (cache_) {
    return cache_->_Free(ptr);
  }
  return free(ptr);
};

SlabCache::SlabCache(size_t obj_size, size_t slab_size,
                     MemoryAllocFree *mem_allocator) {
  obj_size_ = obj_size;
  slab_size_ = slab_size;

  obj_num_ = 0;
  active_obj_num_ = 0;
  slab_empty_num_ = 0;
  slab_num_ = 0;

  if (obj_size <= 0) {
    Abort("object size is invalid.");
  }

  batch_object_num_ = slab_size / obj_size;
  if (batch_object_num_ == 0) {
    Abort("slab size or object size is invalid.");
  }

  ListInit(&full_);
  ListInit(&partial_);
  ListInit(&empty_);

  mem_allocator_ = mem_allocator;
  SlabCacheReclaimer::Instance().AddSlabCache(this);
};

SlabCache::~SlabCache() {
  SlabCacheReclaimer::Instance().RmvSlabCache(this);
  RemoveSlabs(&full_);
  RemoveSlabs(&partial_);
  RemoveSlabs(&empty_);
  slab_empty_num_ = 0;
  mem_allocator_ = nullptr;
};

std::shared_ptr<void> SlabCache::AllocSharedPtr() {
  void *ptr = nullptr;
  Slab *s = nullptr;

  AllocObject(&ptr, &s);
  if (ptr == nullptr) {
    return nullptr;
  }

  std::shared_ptr<void> ret(ptr, [=](void *ptr) { this->FreeObject(ptr, s); });
  return ret;
}

void SlabCache::AllocObject(void **obj, Slab **slab) {
  Slab *s = nullptr;
  bool is_stop = false;
  void *ret = nullptr;
  ListHead *from_list = nullptr;

  std::unique_lock<std::mutex> lock(lock_);
  while (ret == nullptr && is_stop == false) {
    if (!ListEmpty(&partial_)) {
      from_list = &partial_;
    } else if (!ListEmpty(&empty_)) {
      from_list = &empty_;
    } else {
      is_stop = !GrowLocked(&lock);
      continue;
    }

    s = ListFirstEntry(from_list, Slab, list);
    if (s == nullptr) {
      continue;
    }

    ret = s->Alloc();
    if (ret == nullptr) {
      is_stop = !GrowLocked(&lock);
      continue;
    }

    if (from_list == &empty_) {
      if (slab_empty_num_ == 0) {
        Abort("slab number is invalid.");
      }

      slab_empty_num_--;
    }

    if (s->IsFull()) {
      ListDel(&s->list);
      ListAdd(&s->list, &full_);
    } else if (from_list != &partial_) {
      ListDel(&s->list);
      ListAdd(&s->list, &partial_);
    }
  }

  if (ret == nullptr) {
    *obj = nullptr;
    *slab = nullptr;
    return;
  }

  active_obj_num_++;
  *obj = ret;
  *slab = s;
}

void SlabCache::FreeObject(void *obj, Slab *slab) {
  std::unique_lock<std::mutex> lock(lock_);

  active_obj_num_--;
  slab->Free(obj);
  if (slab->IsEmpty()) {
    ListDel(&slab->list);
    ListAdd(&slab->list, &empty_);
    slab_empty_num_++;
  } else if (!slab->IsFull()) {
    ListDel(&slab->list);
    ListAdd(&slab->list, &partial_);
  }
}

void SlabCache::Shrink(int keep, time_t before) {
  size_t shrink_num = 0;
  int empty_number = slab_empty_num_;

  if (empty_number <= 0) {
    return;
  }

  if (keep == 0) {
    shrink_num = ~0;
  } else if (empty_number > keep) {
    shrink_num = slab_empty_num_ - keep;
  }

  RemoveSlabs(&empty_, shrink_num, before);
}

void SlabCache::Reclaim(time_t before) {
  if (obj_num_ == 0 || batch_object_num_ == 0) {
    return;
  }

  const int free_percent_threshold = 10;
  auto free_obj_percent =
      (slab_empty_num_ * batch_object_num_ * 100) / (obj_num_ * 100);
  auto shrink_percent = free_obj_percent * 100 - free_percent_threshold;
  if (shrink_percent <= 0) {
    /* shrink unused slabs */
    RemoveSlabs(&empty_, slab_empty_num_, 60 * 10);
    return;
  }

  int shrink_obj_num = obj_num_ * shrink_percent / 100;
  int shrink_num = shrink_obj_num / batch_object_num_;
  if (shrink_num <= 0) {
    return;
  }

  RemoveSlabs(&empty_, shrink_num, before);
}

uint32_t SlabCache::SlabNumber() {
  std::unique_lock<std::mutex> lock(lock_);
  return slab_num_;
};

int SlabCache::GetEmptySlabNumber() { return slab_empty_num_; }

size_t SlabCache::ObjectSize() { return obj_size_; };

uint32_t SlabCache::GetObjNumber() { return obj_num_; }

uint32_t SlabCache::GetFreeObjNumber() { return obj_num_ - active_obj_num_; }

uint32_t SlabCache::GetActiveObjNumber() { return active_obj_num_; }

void SlabCache::RemoveSlabs(ListHead *head) { RemoveSlabs(head, -1, 0); }

void SlabCache::RemoveSlabLocked(Slab *s) {
  ListDel(&s->list);
  slab_num_--;
  obj_num_ -= s->ObjectNumber();
}

void SlabCache::RemoveSlabs(ListHead *head, size_t count, time_t time_before) {
  Slab *s = nullptr;
  Slab *tmp = nullptr;
  size_t loop_count = count;
  time_t now = time(0);

  ListHead list_free;
  ListInit(&list_free);

  if (count <= 0) {
    return;
  }

  std::unique_lock<std::mutex> lock(lock_);
  ListForEachEntrySafe(s, tmp, head, list) {
    if (s->AliveTime() > (now - time_before)) {
      continue;
    }

    RemoveSlabLocked(s);

    ListAdd(&s->list, &list_free);
    if (head == &empty_) {
      slab_empty_num_--;
    }

    loop_count--;
    if (loop_count <= 0) {
      break;
    }
  }
  lock.unlock();

  ListForEachEntrySafe(s, tmp, &list_free, list) {
    ListDel(&s->list);
    delete s;
  }
}

bool SlabCache::GrowLocked(std::unique_lock<std::mutex> *lock) {
  try {
    lock->unlock();
    Slab *s = new Slab(this, obj_size_, slab_size_);
    if (s->Init() == false) {
      lock->lock();
      delete s;
      return false;
    }
    lock->lock();

    ListAddTail(&s->list, &empty_);
    slab_empty_num_++;
    slab_num_++;
    obj_num_ += s->ObjectNumber();
  } catch (...) {
    return false;
  }
  return true;
}

void *SlabCache::_Alloc(size_t size) {
  if (mem_allocator_) {
    return mem_allocator_->MemAlloc(size);
  }

  return malloc(size);
}

void SlabCache::_Free(void *ptr) {
  if (mem_allocator_) {
    return mem_allocator_->MemFree(ptr);
  }

  return free(ptr);
}

SlabCacheReclaimer &SlabCacheReclaimer::Instance() {
  static SlabCacheReclaimer reclaimer;
  return reclaimer;
}

void SlabCacheReclaimer::AddSlabCache(SlabCache *slabcache) {
  std::unique_lock<std::mutex> lock(cache_lock_);
  if (slab_cache_list_.find(slabcache) != slab_cache_list_.end()) {
    return;
  }

  slab_cache_list_[slabcache] = slabcache;
  lock.unlock();
  StartReclaimWorker();
}

void SlabCacheReclaimer::RmvSlabCache(SlabCache *slabcache) {
  std::unique_lock<std::mutex> lock(cache_lock_);
  auto itr = slab_cache_list_.find(slabcache);
  if (itr == slab_cache_list_.end()) {
    return;
  }
  slab_cache_list_.erase(itr);
  lock.unlock();
  StopReclaimWoker();
}

void SlabCacheReclaimer::DoReclaim() {
  std::unique_lock<std::mutex> lock(cache_lock_);
  for (auto itr : slab_cache_list_) {
    auto slabcache = itr.first;
    slabcache->Reclaim();
  }
}

SlabCacheReclaimer::SlabCacheReclaimer() {}

SlabCacheReclaimer::~SlabCacheReclaimer() {
  if (reclaimer_timer_ == nullptr) {
    return;
  }
  StopReclaimWoker();
}

void SlabCacheReclaimer::StartReclaimWorker() {
  if (slab_cache_num_++ > 0) {
    return;
  }

  timer_.SetName("Slab-Reclaim");
  timer_.Start();
  reclaimer_timer_ = std::make_shared<TimerTask>();
  reclaimer_timer_->Callback(&SlabCacheReclaimer::DoReclaim, this);
  timer_.Schedule(reclaimer_timer_, 0, 10 * 1000);
}

void SlabCacheReclaimer::StopReclaimWoker() {
  if (--slab_cache_num_ > 0) {
    return;
  }

  reclaimer_timer_->Stop();
  reclaimer_timer_ = nullptr;
  timer_.Stop();
}

}  // namespace modelbox
