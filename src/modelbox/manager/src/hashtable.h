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


#ifndef _hashtable_
#define _hashtable_

#include <stdbool.h>
#include <stdint.h>

#include "list.h"

#ifndef __WORDSIZE
#define __WORDSIZE (__SIZEOF_LONG__ * 8)
#endif

#ifndef BITS_PER_LONG
#define BITS_PER_LONG __WORDSIZE
#endif

#ifndef __same_type
#define __same_type(a, b) __builtin_types_compatible_p(typeof(a), typeof(b))
#endif
#define MANAGER_BUILD_BUG_ON_ZERO(e) (sizeof(struct { int : -!!(e); }))
#define __must_be_array(a) MANAGER_BUILD_BUG_ON_ZERO(__same_type((a), &(a)[0]))
#define MANAGER_ARRAY_SIZE(arr) \
  (sizeof(arr) / sizeof((arr)[0]) + __must_be_array(arr))

#define MANAGER_DEFINE_HASHTABLE(name, bits)                           \
  struct hlist_head name[1 << (bits)] = {[0 ...((1 << (bits)) - 1)] = \
                                             MANAGER_HLIST_HEAD_INIT}

#define MANAGER_DECLARE_HASHTABLE(name, bits) struct hlist_head name[1 << (bits)]

#define MANAGER_HASH_SIZE(name) (MANAGER_ARRAY_SIZE(name))
#define MANAGER_HASH_BITS(name) ilog2(MANAGER_HASH_SIZE(name))

static inline int fls(unsigned int x) { return 32 - __builtin_clz(x); }

static inline int fls64(unsigned long x) { return 64 - __builtin_clzll(x); }

static inline int __ilog2_u32(uint32_t n) { return fls(n) - 1; }

static inline int __ilog2_u64(uint64_t n) { return fls64(n) - 1; }

/* clang-format off */
#define ilog2(n)				\
(						\
	__builtin_constant_p(n) ? (		\
		(n) < 2 ? 0 :			\
		(n) & (1ULL << 63) ? 63 :	\
		(n) & (1ULL << 62) ? 62 :	\
		(n) & (1ULL << 61) ? 61 :	\
		(n) & (1ULL << 60) ? 60 :	\
		(n) & (1ULL << 59) ? 59 :	\
		(n) & (1ULL << 58) ? 58 :	\
		(n) & (1ULL << 57) ? 57 :	\
		(n) & (1ULL << 56) ? 56 :	\
		(n) & (1ULL << 55) ? 55 :	\
		(n) & (1ULL << 54) ? 54 :	\
		(n) & (1ULL << 53) ? 53 :	\
		(n) & (1ULL << 52) ? 52 :	\
		(n) & (1ULL << 51) ? 51 :	\
		(n) & (1ULL << 50) ? 50 :	\
		(n) & (1ULL << 49) ? 49 :	\
		(n) & (1ULL << 48) ? 48 :	\
		(n) & (1ULL << 47) ? 47 :	\
		(n) & (1ULL << 46) ? 46 :	\
		(n) & (1ULL << 45) ? 45 :	\
		(n) & (1ULL << 44) ? 44 :	\
		(n) & (1ULL << 43) ? 43 :	\
		(n) & (1ULL << 42) ? 42 :	\
		(n) & (1ULL << 41) ? 41 :	\
		(n) & (1ULL << 40) ? 40 :	\
		(n) & (1ULL << 39) ? 39 :	\
		(n) & (1ULL << 38) ? 38 :	\
		(n) & (1ULL << 37) ? 37 :	\
		(n) & (1ULL << 36) ? 36 :	\
		(n) & (1ULL << 35) ? 35 :	\
		(n) & (1ULL << 34) ? 34 :	\
		(n) & (1ULL << 33) ? 33 :	\
		(n) & (1ULL << 32) ? 32 :	\
		(n) & (1ULL << 31) ? 31 :	\
		(n) & (1ULL << 30) ? 30 :	\
		(n) & (1ULL << 29) ? 29 :	\
		(n) & (1ULL << 28) ? 28 :	\
		(n) & (1ULL << 27) ? 27 :	\
		(n) & (1ULL << 26) ? 26 :	\
		(n) & (1ULL << 25) ? 25 :	\
		(n) & (1ULL << 24) ? 24 :	\
		(n) & (1ULL << 23) ? 23 :	\
		(n) & (1ULL << 22) ? 22 :	\
		(n) & (1ULL << 21) ? 21 :	\
		(n) & (1ULL << 20) ? 20 :	\
		(n) & (1ULL << 19) ? 19 :	\
		(n) & (1ULL << 18) ? 18 :	\
		(n) & (1ULL << 17) ? 17 :	\
		(n) & (1ULL << 16) ? 16 :	\
		(n) & (1ULL << 15) ? 15 :	\
		(n) & (1ULL << 14) ? 14 :	\
		(n) & (1ULL << 13) ? 13 :	\
		(n) & (1ULL << 12) ? 12 :	\
		(n) & (1ULL << 11) ? 11 :	\
		(n) & (1ULL << 10) ? 10 :	\
		(n) & (1ULL <<  9) ?  9 :	\
		(n) & (1ULL <<  8) ?  8 :	\
		(n) & (1ULL <<  7) ?  7 :	\
		(n) & (1ULL <<  6) ?  6 :	\
		(n) & (1ULL <<  5) ?  5 :	\
		(n) & (1ULL <<  4) ?  4 :	\
		(n) & (1ULL <<  3) ?  3 :	\
		(n) & (1ULL <<  2) ?  2 :	\
		1 ) :				\
	(sizeof(n) <= 4) ?			\
	__ilog2_u32(n) :			\
	__ilog2_u64(n)				\
 )
/* clang-format on */

#define GOLDEN_RATIO_32 0x61C88647
#define GOLDEN_RATIO_64 0x61C8864680B583EBull

static inline uint32_t __hash_32(uint32_t val) { return val * GOLDEN_RATIO_32; }

static inline uint32_t hash_32(uint32_t val, unsigned int bits) {
  /* High bits are more random, so use them. */
  return __hash_32(val) >> (32 - bits);
}

static inline uint32_t hash_long(uint64_t val, unsigned int bits) {
#if BITS_PER_LONG == 64
  /* 64x64-bit multiply is efficient on all 64-bit processors */
  return val * GOLDEN_RATIO_64 >> (64 - bits);
#else
  /* Hash 64 bits using only 32x32-bit multiply. */
  return hash_32((uint32_t)val ^ __hash_32(val >> 32), bits);
#endif
}

static inline uint32_t hash_ptr(const void *ptr, unsigned int bits) {
  return hash_long((unsigned long)ptr, bits);
}

static inline uint32_t hash32_ptr(const void *ptr) {
  unsigned long val = (unsigned long)ptr;

#if BITS_PER_LONG == 64
  val ^= (val >> 32);
#endif
  return (uint32_t)val;
}

static inline unsigned long hash_string(const char *str) {
  unsigned long v = 0;
  const char *c;
  for (c = str; *c;) v = (((v << 1) + (v >> 14)) ^ (*c++)) & 0x3fff;
  return (v);
}

/* Use hash_32 when possible to allow for fast 32bit hashing in 64bit kernels.
 */
#define hash_min(val, bits) \
  (sizeof(val) <= 4 ? hash_32(val, bits) : hash_long(val, bits))

static inline void __hash_init(struct hlist_head *ht, unsigned int sz) {
  unsigned int i;

  for (i = 0; i < sz; i++) MANAGER_INIT_HLIST_HEAD(&ht[i]);
}

#define MANAGER_JHASH_INITVAL 0xdeadbeef
/* Best hash sizes are of power of two */
#define jhash_size(n) ((uint32_t)1 << (n))
/* Mask the hash value, i.e (value & jhash_mask(n)) instead of (value % n) */
#define jhash_mask(n) (jhash_size(n) - 1)

static inline uint32_t rol32(uint32_t word, unsigned int shift) {
  return (word << shift) | (word >> ((-shift) & 31));
}

/* __jhash_mix -- mix 3 32-bit values reversibly. */
#define __jhash_mix(a, b, c) \
  {                          \
    a -= c;                  \
    a ^= rol32(c, 4);        \
    c += b;                  \
    b -= a;                  \
    b ^= rol32(a, 6);        \
    a += c;                  \
    c -= b;                  \
    c ^= rol32(b, 8);        \
    b += a;                  \
    a -= c;                  \
    a ^= rol32(c, 16);       \
    c += b;                  \
    b -= a;                  \
    b ^= rol32(a, 19);       \
    a += c;                  \
    c -= b;                  \
    c ^= rol32(b, 4);        \
    b += a;                  \
  }

/* __jhash_final - final mixing of 3 32-bit values (a,b,c) into c */
#define __jhash_final(a, b, c) \
  {                            \
    c ^= b;                    \
    c -= rol32(b, 14);         \
    a ^= c;                    \
    a -= rol32(c, 11);         \
    b ^= a;                    \
    b -= rol32(a, 25);         \
    c ^= b;                    \
    c -= rol32(b, 16);         \
    a ^= c;                    \
    a -= rol32(c, 4);          \
    b ^= a;                    \
    b -= rol32(a, 14);         \
    c ^= b;                    \
    c -= rol32(b, 24);         \
  }

struct __una_u32 {
  uint32_t x;
} __attribute__((__packed__));

static inline uint32_t __get_unaligned_cpu32(const void *p) {
  const struct __una_u32 *ptr = (const struct __una_u32 *)p;
  return ptr->x;
}

static inline uint32_t jhash(const void *key, uint32_t length,
                             uint32_t initval) {
  uint32_t a, b, c;
  const uint8_t *k = (uint8_t *)key;

  /* Set up the internal state */
  a = b = c = MANAGER_JHASH_INITVAL + length + initval;

  /* All but the last block: affect some 32 bits of (a,b,c) */
  while (length > 12) {
    a += __get_unaligned_cpu32(k);
    b += __get_unaligned_cpu32(k + 4);
    c += __get_unaligned_cpu32(k + 8);
    __jhash_mix(a, b, c);
    length -= 12;
    k += 12;
  }
  /* Last block: affect all 32 bits of (c) */
  /* All the case statements fall through */
  switch (length) {
    case 12:
      c += (uint32_t)k[11] << 24;
    case 11:
      c += (uint32_t)k[10] << 16;
    case 10:
      c += (uint32_t)k[9] << 8;
    case 9:
      c += k[8];
    case 8:
      b += (uint32_t)k[7] << 24;
    case 7:
      b += (uint32_t)k[6] << 16;
    case 6:
      b += (uint32_t)k[5] << 8;
    case 5:
      b += k[4];
    case 4:
      a += (uint32_t)k[3] << 24;
    case 3:
      a += (uint32_t)k[2] << 16;
    case 2:
      a += (uint32_t)k[1] << 8;
    case 1:
      a += k[0];
      __jhash_final(a, b, c);
    case 0: /* Nothing left to add */
      break;
  }

  return c;
}

#define hash_init(hashtable) __hash_init(hashtable, MANAGER_HASH_SIZE(hashtable))

#define hash_add(hashtable, node, key) \
  hlist_add_head(node, &hashtable[hash_min(key, MANAGER_HASH_BITS(hashtable))])

static inline bool hash_hashed(struct hlist_node *node) {
  return !hlist_unhashed(node);
}

static inline bool __hash_empty(struct hlist_head *ht, unsigned int sz) {
  unsigned int i;

  for (i = 0; i < sz; i++)
    if (!hlist_empty(&ht[i])) return false;

  return true;
}

#define hash_empty(hashtable) \
  __hash_empty(hashtable, MANAGER_HASH_SIZE(hashtable))

static inline void hash_del(struct hlist_node *node) { hlist_del_init(node); }

#define hash_for_each(name, bkt, obj, member)                                \
  for ((bkt) = 0, obj = NULL; obj == NULL && (bkt) < MANAGER_HASH_SIZE(name); \
       (bkt)++)                                                              \
  hlist_for_each_entry(obj, &name[bkt], member)

#define hash_for_each_safe(name, bkt, tmp, obj, member)                      \
  for ((bkt) = 0, obj = NULL; obj == NULL && (bkt) < MANAGER_HASH_SIZE(name); \
       (bkt)++)                                                              \
  hlist_for_each_entry_safe(obj, tmp, &name[bkt], member)

#define hash_for_each_possible(name, obj, member, key)                    \
  hlist_for_each_entry(obj, &name[hash_min(key, MANAGER_HASH_BITS(name))], \
                       member)

#define hash_for_each_possible_safe(name, obj, tmp, member, key) \
  hlist_for_each_entry_safe(                                     \
      obj, tmp, &name[hash_min(key, MANAGER_HASH_BITS(name))], member)

#endif
