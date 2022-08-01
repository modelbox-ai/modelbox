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


#ifndef LIST_H
#define LIST_H

#ifdef __cplusplus
extern "C" {
#define LIST_TYPEOF __typeof__
#else
#define LIST_TYPEOF typeof
#endif /* __cplusplus */

#define LIST_OFFSET_OF(type, member) ((size_t) & ((type *)0)->member)
#define LIST_CONTAINER_OF(ptr, type, member)                 \
  ({                                                         \
    const LIST_TYPEOF(((type *)0)->member) *__mptr = (ptr);  \
    (type *)((char *)__mptr - LIST_OFFSET_OF(type, member)); \
  })

#define LIST_POISON1 ((void *)0x00100100)
#define LIST_POISON2 ((void *)0x00200200)

/**
 * @brief list head
 */
typedef struct _ListHead {
  /// @brief next list head
  struct _ListHead *next;
  /// @brief prev list head
  struct _ListHead *prev;
} ListHead;

/* Init list head */
#define LIST_HEAD_INIT(name) \
  { &(name), &(name) }

/* define a list */
#define LIST_HEAD(name) LIST_HEAD name = LIST_HEAD_INIT(name)

/*Init list head*/
static inline void ListInit(ListHead *list) {
  list->next = list;
  list->prev = list;
}

static inline void _ListAdd(ListHead *newnode, ListHead *prev, ListHead *next) {
  next->prev = newnode;
  newnode->next = next;
  newnode->prev = prev;
  prev->next = newnode;
}

/* Add new item to list head */
static inline void ListAdd(ListHead *newnode, ListHead *head) {
  _ListAdd(newnode, head, head->next);
}

/* Add new item to list tail */
static inline void ListAddTail(ListHead *newnode, ListHead *head) {
  _ListAdd(newnode, head->prev, head);
}

static inline void _ListDel(ListHead *prev, ListHead *next) {
  next->prev = prev;
  prev->next = next;
}

/* delete a item from list */
static inline void ListDel(ListHead *entry) {
  _ListDel(entry->prev, entry->next);
  entry->next = (ListHead *)LIST_POISON1;
  entry->prev = (ListHead *)LIST_POISON2;
}

/* delete a item from list and init the item */
static inline void ListDelInit(ListHead *entry) {
  _ListDel(entry->prev, entry->next);
  ListInit(entry);
}

/* set item invalid */
static inline void ListInitEntry(ListHead *entry) {
  entry->next = (ListHead *)LIST_POISON1;
  entry->prev = (ListHead *)LIST_POISON2;
}

/* check item is in list */
static inline int ListEntryNotInList(ListHead *entry) {
  return ((entry->next == (ListHead *)LIST_POISON1) &&
          (entry->prev == (ListHead *)LIST_POISON2));
}

/* is list empty */
static inline int ListEmpty(const ListHead *head) { return head->next == head; }

/* is list empty */
static inline int ListEmptyCareful(const ListHead *head) {
  ListHead *next = head->next;
  return (next == head) && (next == head->prev);
}

/* Get list entry */
#define ListEntry(ptr, type, member) LIST_CONTAINER_OF(ptr, type, member)

/* Get First Entry */
#define ListFirstEntry(ptr, type, member) ListEntry((ptr)->next, type, member)

/* Get Last Entry */
#define ListLastEntry(ptr, type, member) ListEntry((ptr)->prev, type, member)

/* iterator the list */
#define ListForEachEntry(pos, head, member)                          \
  for ((pos) = ListEntry((head)->next, LIST_TYPEOF(*(pos)), member); \
       &(pos)->member != (head);                                     \
       (pos) = ListEntry((pos)->member.next, LIST_TYPEOF(*(pos)), member))

/* iterator the list */
#define ListForEachEntrySafe(pos, n, head, member)                      \
  for ((pos) = ListEntry((head)->next, LIST_TYPEOF(*(pos)), member),    \
      (n) = ListEntry((pos)->member.next, LIST_TYPEOF(*(pos)), member); \
       &(pos)->member != (head); (pos) = (n),                           \
      (n) = ListEntry((n)->member.next, LIST_TYPEOF(*(n)), member))

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
