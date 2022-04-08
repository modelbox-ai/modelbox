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

#ifndef MODELBOX_MEMORY_STUB_H_
#define MODELBOX_MEMORY_STUB_H_

#include <memory>

#ifndef ANDROID

#define modelbox_dynamic_pointer_cast std::dynamic_pointer_cast

#else /* ANDROID */

#define modelbox_dynamic_pointer_cast modelbox::dynamic_pointer_cast

namespace modelbox {

/* LLVM __dynamic_cast calls is_equal() with use_strcmp=false,
 * so the string names are not compared. It will return NULL
 * if the object is allocated in one so file and dynamicly casted
 * in another so file. */
template<class _Tp, class _Up>
inline _LIBCPP_INLINE_VISIBILITY
typename std::enable_if
<
    !std::is_array<_Tp>::value && !std::is_array<_Up>::value,
    std::shared_ptr<_Tp>
>::type
dynamic_pointer_cast(const std::shared_ptr<_Up>& __r) _NOEXCEPT
{
    auto __p = std::dynamic_pointer_cast<_Tp>(__r);
    if (__p)  {
        return __p;
    }
    return std::static_pointer_cast<_Tp>(__r);
}

}  // namespace modelbox
#endif  /* ANDROID */

#endif /* MODELBOX_MEMORY_STUB_H_ */
