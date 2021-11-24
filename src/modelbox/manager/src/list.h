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


#ifndef MANAGER_LIST_H
#define MANAGER_LIST_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

struct list_head {
  struct list_head *next, *prev;
};

#define MANAGER_LIST_HEAD_INIT(name) \
  { &(name), &(name) }

#define MANAGER_LIST_HEAD(name) \
  struct list_head name = MANAGER_LIST_HEAD_INIT(name)

static inline void MANAGER_INIT_LIST_HEAD(struct list_head *list) {
  list->next = list;
  list->prev = list;
}

static inline void __list_add(struct list_head *add, struct list_head *prev,
                              struct list_head *next) {
  next->prev = add;
  add->next = next;
  add->prev = prev;
  prev->next = add;
}

static inline void list_add(struct list_head *add, struct list_head *head) {
  __list_add(add, head, head->next);
}

static inline void list_add_tail(struct list_head *add,
                                 struct list_head *head) {
  __list_add(add, head->prev, head);
}

static inline void __list_del(struct list_head *prev, struct list_head *next) {
  next->prev = prev;
  prev->next = next;
}

static inline void list_del(struct list_head *entry) {
  __list_del(entry->prev, entry->next);
  entry->next = NULL;
  entry->prev = NULL;
}

static inline void list_replace(struct list_head *old, struct list_head *add) {
  add->next = old->next;
  add->next->prev = add;
  add->prev = old->prev;
  add->prev->next = add;
}

static inline void list_replace_init(struct list_head *old,
                                     struct list_head *add) {
  list_replace(old, add);
  MANAGER_INIT_LIST_HEAD(old);
}

static inline void list_del_init(struct list_head *entry) {
  __list_del(entry->prev, entry->next);
  MANAGER_INIT_LIST_HEAD(entry);
}

static inline void list_move(struct list_head *list, struct list_head *head) {
  __list_del(list->prev, list->next);
  list_add(list, head);
}

static inline void list_move_tail(struct list_head *list,
                                  struct list_head *head) {
  __list_del(list->prev, list->next);
  list_add_tail(list, head);
}

static inline int list_is_last(const struct list_head *list,
                               const struct list_head *head) {
  return list->next == head;
}

static inline int list_empty(const struct list_head *head) {
  return head->next == head;
}

static inline int list_is_singular(const struct list_head *head) {
  return !list_empty(head) && (head->next == head->prev);
}

#ifndef offsetof
#define offsetof(TYPE, MEMBER) ((size_t) & ((TYPE *)0)->MEMBER)
#endif

#define container_of(ptr, type, member)                \
  ({                                                   \
    const typeof(((type *)0)->member) *__mptr = (ptr); \
    (type *)((char *)__mptr - offsetof(type, member)); \
  })

#define prefetch(x) __builtin_prefetch(x)

#define list_entry(ptr, type, member) container_of(ptr, type, member)

#define list_first_entry(ptr, type, member) \
  list_entry((ptr)->next, type, member)

#define list_for_each(pos, head) \
  for (pos = (head)->next; (pos->next), pos != (head); pos = pos->next)

#define list_for_each_safe(pos, n, head) \
  for (pos = (head)->next, n = pos->next; pos != (head); pos = n, n = pos->next)

#define list_for_each_entry(pos, head, member)               \
  for (pos = list_entry((head)->next, typeof(*pos), member); \
       prefetch(pos->member.next), &pos->member != (head);   \
       pos = list_entry(pos->member.next, typeof(*pos), member))

#define list_for_each_entry_safe(pos, n, head, member)        \
  for (pos = list_entry((head)->next, typeof(*pos), member),  \
      n = list_entry(pos->member.next, typeof(*pos), member); \
       &pos->member != (head);                                \
       pos = n, n = list_entry(n->member.next, typeof(*n), member))

struct hlist_head {
  struct hlist_node *first;
};

struct hlist_node {
  struct hlist_node *next, **pprev;
};

#define MANAGER_HLIST_HEAD_INIT \
  { .first = NULL }
#define MANAGER_HLIST_HEAD(name) struct hlist_head name = {.first = NULL}
#define MANAGER_INIT_HLIST_HEAD(ptr) ((ptr)->first = NULL)
static inline void MANAGER_INIT_HLIST_NODE(struct hlist_node *h) {
  h->next = NULL;
  h->pprev = NULL;
}

static inline int hlist_unhashed(const struct hlist_node *h) {
  return !h->pprev;
}

static inline int hlist_empty(const struct hlist_head *h) { return !h->first; }

static inline void __hlist_del(struct hlist_node *n) {
  struct hlist_node *next = n->next;
  struct hlist_node **pprev = n->pprev;

  *pprev = next;
  if (next) next->pprev = pprev;
}

static inline void hlist_del(struct hlist_node *n) {
  __hlist_del(n);
  n->next = (struct hlist_node *)NULL;
  n->pprev = (struct hlist_node **)NULL;
}

static inline void hlist_del_init(struct hlist_node *n) {
  if (!hlist_unhashed(n)) {
    __hlist_del(n);
    MANAGER_INIT_HLIST_NODE(n);
  }
}

static inline void hlist_add_head(struct hlist_node *n, struct hlist_head *h) {
  struct hlist_node *first = h->first;
  n->next = first;
  if (first) first->pprev = &n->next;
  h->first = n;
  n->pprev = &h->first;
}

#define hlist_entry(ptr, type, member) container_of(ptr, type, member)

#define hlist_for_each(pos, head) \
  for (pos = (head)->first; pos; pos = pos->next)

#define hlist_for_each_safe(pos, n, head)    \
  for (pos = (head)->first; pos && ({        \
                              n = pos->next; \
                              1;             \
                            });              \
       pos = n)

#define hlist_entry_safe(ptr, type, member)              \
  ({                                                     \
    typeof(ptr) ____ptr = (ptr);                         \
    ____ptr ? hlist_entry(____ptr, type, member) : NULL; \
  })

/**
 * hlist_for_each_entry	- iterate over list of given type
 * @pos:	the type * to use as a loop cursor.
 * @head:	the head for your list.
 * @member:	the name of the hlist_node within the struct.
 */
#define hlist_for_each_entry(pos, head, member)                            \
  for (pos = hlist_entry_safe((head)->first, typeof(*(pos)), member); pos; \
       pos = hlist_entry_safe((pos)->member.next, typeof(*(pos)), member))

/**
 * hlist_for_each_entry_continue - iterate over a hlist continuing after current
 * point
 * @pos:	the type * to use as a loop cursor.
 * @member:	the name of the hlist_node within the struct.
 */
#define hlist_for_each_entry_continue(pos, member)                         \
  for (pos = hlist_entry_safe((pos)->member.next, typeof(*(pos)), member); \
       pos;                                                                \
       pos = hlist_entry_safe((pos)->member.next, typeof(*(pos)), member))

/**
 * hlist_for_each_entry_from - iterate over a hlist continuing from current
 * point
 * @pos:	the type * to use as a loop cursor.
 * @member:	the name of the hlist_node within the struct.
 */
#define hlist_for_each_entry_from(pos, member) \
  for (; pos;                                  \
       pos = hlist_entry_safe((pos)->member.next, typeof(*(pos)), member))

/**
 * hlist_for_each_entry_safe - iterate over list of given type safe against
 * removal of list entry
 * @pos:	the type * to use as a loop cursor.
 * @n:		another &struct hlist_node to use as temporary storage
 * @head:	the head for your list.
 * @member:	the name of the hlist_node within the struct.
 */
#define hlist_for_each_entry_safe(pos, n, head, member)             \
  for (pos = hlist_entry_safe((head)->first, typeof(*pos), member); \
       pos && ({                                                    \
         n = pos->member.next;                                      \
         1;                                                         \
       });                                                          \
       pos = hlist_entry_safe(n, typeof(*pos), member))

#ifdef __cplusplus
}
#endif /*__cplusplus */
#endif
