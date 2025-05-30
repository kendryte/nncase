/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#if defined(_MSC_VER)
// Fix: https://learn.microsoft.com/en-us/cpp/cpp/empty-bases
#define NTT_EMPTY_BASES __declspec(empty_bases)
#else
#define NTT_EMPTY_BASES
#endif

#ifdef _MSC_VER
#define NTT_ASSUME(...)
#define NTT_UNREACHABLE() __assume(0)
#define NTT_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#elif __GNUC__
#define NTT_ASSUME(...)                                                        \
    do {                                                                       \
        if (!(__VA_ARGS__))                                                    \
            __builtin_unreachable();                                           \
    } while (0)
#define NTT_UNREACHABLE() __builtin_unreachable()
#define NTT_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define NTT_ASSUME(...) __builtin_assume(__VA_ARGS__)
#define NTT_UNREACHABLE() __builtin_unreachable()
#define NTT_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif
