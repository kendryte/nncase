// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#define MULTI_CORE_CPU
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;

namespace Nncase.CodeGen.CPU;

public static class CSourceBuiltn
{
#if MULTI_CORE_CPU
    public static string ClusterDef()
    {
        return @"#pragma once
#include <hardware_context.h>
#include <tdma.h>

namespace block0 {
#include ""shared.h""
constexpr size_t bid = 0;
namespace thread0 {
constexpr size_t tid = 0;
#include ""kernel.h""
} // namespace thread0
namespace thread1 {
constexpr size_t tid = 1;
#include ""kernel.h""
} // namespace thread1
namespace thread2 {
constexpr size_t tid = 2;
#include ""kernel.h""
} // namespace thread2
namespace thread3 {
constexpr size_t tid = 3;
#include ""kernel.h""
} // namespace thread3
} // namespace block0

namespace block1 {
#include ""shared.h""
constexpr size_t bid = 1;
namespace thread0 {
constexpr size_t tid = 0;
#include ""kernel.h""
} // namespace thread0
namespace thread1 {
constexpr size_t tid = 1;
#include ""kernel.h""
} // namespace thread1
namespace thread2 {
constexpr size_t tid = 2;
#include ""kernel.h""
} // namespace thread2
namespace thread3 {
constexpr size_t tid = 3;
#include ""kernel.h""
} // namespace thread3
} // namespace block1

namespace block2 {
#include ""shared.h""
constexpr size_t bid = 2;
namespace thread0 {
constexpr size_t tid = 0;
#include ""kernel.h""
} // namespace thread0
namespace thread1 {
constexpr size_t tid = 1;
#include ""kernel.h""
} // namespace thread1
namespace thread2 {
constexpr size_t tid = 2;
#include ""kernel.h""
} // namespace thread2
namespace thread3 {
constexpr size_t tid = 3;
#include ""kernel.h""
} // namespace thread3
} // namespace block2

namespace block3 {
#include ""shared.h""
constexpr size_t bid = 3;
namespace thread0 {
constexpr size_t tid = 0;
#include ""kernel.h""
} // namespace thread0
namespace thread1 {
constexpr size_t tid = 1;
#include ""kernel.h""
} // namespace thread1
namespace thread2 {
constexpr size_t tid = 2;
#include ""kernel.h""
} // namespace thread2
namespace thread3 {
constexpr size_t tid = 3;
#include ""kernel.h""
} // namespace thread3
} // namespace block3

namespace block4 {
#include ""shared.h""
constexpr size_t bid = 4;
namespace thread0 {
constexpr size_t tid = 0;
#include ""kernel.h""
} // namespace thread0
namespace thread1 {
constexpr size_t tid = 1;
#include ""kernel.h""
} // namespace thread1
namespace thread2 {
constexpr size_t tid = 2;
#include ""kernel.h""
} // namespace thread2
namespace thread3 {
constexpr size_t tid = 3;
#include ""kernel.h""
} // namespace thread3
} // namespace block4

namespace block5 {
#include ""shared.h""
constexpr size_t bid = 5;
namespace thread0 {
constexpr size_t tid = 0;
#include ""kernel.h""
} // namespace thread0
namespace thread1 {
constexpr size_t tid = 1;
#include ""kernel.h""
} // namespace thread1
namespace thread2 {
constexpr size_t tid = 2;
#include ""kernel.h""
} // namespace thread2
namespace thread3 {
constexpr size_t tid = 3;
#include ""kernel.h""
} // namespace thread3
} // namespace block5

namespace block6 {
#include ""shared.h""
constexpr size_t bid = 6;
namespace thread0 {
constexpr size_t tid = 0;
#include ""kernel.h""
} // namespace thread0
namespace thread1 {
constexpr size_t tid = 1;
#include ""kernel.h""
} // namespace thread1
namespace thread2 {
constexpr size_t tid = 2;
#include ""kernel.h""
} // namespace thread2
namespace thread3 {
constexpr size_t tid = 3;
#include ""kernel.h""
} // namespace thread3
} // namespace block6

namespace block7 {
#include ""shared.h""
constexpr size_t bid = 7;
namespace thread0 {
constexpr size_t tid = 0;
#include ""kernel.h""
} // namespace thread0
namespace thread1 {
constexpr size_t tid = 1;
#include ""kernel.h""
} // namespace thread1
namespace thread2 {
constexpr size_t tid = 2;
#include ""kernel.h""
} // namespace thread2
namespace thread3 {
constexpr size_t tid = 3;
#include ""kernel.h""
} // namespace thread3
} // namespace block7
";
    }

    public static string CMakeDef(string name, [CallerFilePath] string? callerFilePath = null)
    {
        var solutionPath = Path.GetFullPath(Path.Join(callerFilePath!, "../../../.."));
        return $@"project(cpu)
cmake_minimum_required(VERSION 3.13)

add_library(
  cpu_cmodel STATIC
  {solutionPath}/modules/cpu/src/runtime/cmodel/src/dummy.cpp
  {solutionPath}/modules/cpu/src/runtime/hardware_context.cpp)

set(CMAKE_CXX_STANDARD 20)

target_include_directories(
  cpu_cmodel
  PUBLIC
    {solutionPath}/modules/cpu/src/runtime/cmodel/include
)
target_include_directories(
  cpu_cmodel
  PUBLIC {solutionPath}/modules/cpu/include)
target_include_directories(
  cpu_cmodel
  PUBLIC {solutionPath}/modules/cpu/src/runtime/)
set_target_properties(cpu_cmodel PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL ""Release"")
  add_compile_options(-O1)
endif()
add_compile_options(-fPIE)
add_link_options(
  -no-pie
  -nostartfiles
  -fPIC
  -fno-stack-protector
  -static
  -Wl,-e,_Z6_startPN6nncase7runtime3cpu19hardware_context_mtEPNS1_15runtime_util_mtEPNS1_19nncase_method_tableEPPh
)
add_executable({name} ""main.cpp"")
target_link_libraries({name} cpu_cmodel)
";
    }

    public const string KernelHeader = @"#include ""thread_context.h""
using namespace shared;
";

    public static string MakeMain(TIR.PrimFunction primFunction)
    {
        string device_tensors = string.Join('\n', primFunction.Parameters.AsValueEnumerable().Select(b => $"static tensor<{b.ElemType.ToC()},{b.MemSpan.Location.ToC()}> *{b.Name};").ToArray());

        string init_tensors = string.Join('\n', primFunction.Parameters.ToArray().Select((b, i) =>
        {
            var size = TensorUtilities.GetSize(b.CheckedShape.ToValueArray(), TensorUtilities.GetStrides(b.CheckedShape.ToValueArray()), 1);
            return $@"auto {b.Name}_ = tensor<{b.ElemType.ToC()}, {b.MemSpan.Location.ToC()}>(gsl::make_span(({b.ElemType.ToC()}*)inputs[{i}], {size}), {{{string.Join(',', b.CheckedShape)}}});
    {b.Name} = &{b.Name}_;";
        }));

        return @$"#include ""cluster_def.h""
#include <runtime_utils.h>

#define DEFINE_TFUNC(b, t)                                                     \
    void *f_##b##_##t(void *arg) {{                                             \
        block##b::thread##t::{primFunction.Name}({string.Join(',', primFunction.Parameters.AsValueEnumerable().Select(b => '*' + b.Name).ToArray())});               \
        return arg;                                                            \
    }}

#define DEFINE_BFUNC(b)                                                        \
    DEFINE_TFUNC(b, 0)                                                         \
    DEFINE_TFUNC(b, 1)                                                         \
    DEFINE_TFUNC(b, 2)                                                         \
    DEFINE_TFUNC(b, 3)

{device_tensors}

DEFINE_BFUNC(0)
DEFINE_BFUNC(1)
DEFINE_BFUNC(2)
DEFINE_BFUNC(3)
DEFINE_BFUNC(4)
DEFINE_BFUNC(5)
DEFINE_BFUNC(6)
DEFINE_BFUNC(7)

void _start(hardware_context_mt *hw_ctx_impl, runtime_util_mt *rt_util_mt,
            nncase_mt_t *nncase_mt_impl, uint8_t **inputs) {{global_hardware_init(hw_ctx_impl);
    runtime_util = rt_util_mt;
    nncase_mt = nncase_mt_impl;

    {init_tensors}

    pthread_t t_0_0, t_1_0, t_2_0, t_3_0, t_4_0, t_5_0, t_6_0, t_7_0;
    pthread_t t_0_1, t_1_1, t_2_1, t_3_1, t_4_1, t_5_1, t_6_1, t_7_1;
    pthread_t t_0_2, t_1_2, t_2_2, t_3_2, t_4_2, t_5_2, t_6_2, t_7_2;
    pthread_t t_0_3, t_1_3, t_2_3, t_3_3, t_4_3, t_5_3, t_6_3, t_7_3;

    runtime_util->create_thread(t_0_0, NULL, f_0_0);
    runtime_util->create_thread(t_0_1, NULL, f_0_1);
    runtime_util->create_thread(t_0_2, NULL, f_0_2);
    runtime_util->create_thread(t_0_3, NULL, f_0_3);
    runtime_util->create_thread(t_1_0, NULL, f_1_0);
    runtime_util->create_thread(t_1_1, NULL, f_1_1);
    runtime_util->create_thread(t_1_2, NULL, f_1_2);
    runtime_util->create_thread(t_1_3, NULL, f_1_3);
    runtime_util->create_thread(t_2_0, NULL, f_2_0);
    runtime_util->create_thread(t_2_1, NULL, f_2_1);
    runtime_util->create_thread(t_2_2, NULL, f_2_2);
    runtime_util->create_thread(t_2_3, NULL, f_2_3);
    runtime_util->create_thread(t_3_0, NULL, f_3_0);
    runtime_util->create_thread(t_3_1, NULL, f_3_1);
    runtime_util->create_thread(t_3_2, NULL, f_3_2);
    runtime_util->create_thread(t_3_3, NULL, f_3_3);
    runtime_util->create_thread(t_4_0, NULL, f_4_0);
    runtime_util->create_thread(t_4_1, NULL, f_4_1);
    runtime_util->create_thread(t_4_2, NULL, f_4_2);
    runtime_util->create_thread(t_4_3, NULL, f_4_3);
    runtime_util->create_thread(t_5_0, NULL, f_5_0);
    runtime_util->create_thread(t_5_1, NULL, f_5_1);
    runtime_util->create_thread(t_5_2, NULL, f_5_2);
    runtime_util->create_thread(t_5_3, NULL, f_5_3);
    runtime_util->create_thread(t_6_0, NULL, f_6_0);
    runtime_util->create_thread(t_6_1, NULL, f_6_1);
    runtime_util->create_thread(t_6_2, NULL, f_6_2);
    runtime_util->create_thread(t_6_3, NULL, f_6_3);
    runtime_util->create_thread(t_7_0, NULL, f_7_0);
    runtime_util->create_thread(t_7_1, NULL, f_7_1);
    runtime_util->create_thread(t_7_2, NULL, f_7_2);
    runtime_util->create_thread(t_7_3, NULL, f_7_3);

    runtime_util->join_thread(t_0_0);
    runtime_util->join_thread(t_0_1);
    runtime_util->join_thread(t_0_2);
    runtime_util->join_thread(t_0_3);
    runtime_util->join_thread(t_1_0);
    runtime_util->join_thread(t_1_1);
    runtime_util->join_thread(t_1_2);
    runtime_util->join_thread(t_1_3);
    runtime_util->join_thread(t_2_0);
    runtime_util->join_thread(t_2_1);
    runtime_util->join_thread(t_2_2);
    runtime_util->join_thread(t_2_3);
    runtime_util->join_thread(t_3_0);
    runtime_util->join_thread(t_3_1);
    runtime_util->join_thread(t_3_2);
    runtime_util->join_thread(t_3_3);
    runtime_util->join_thread(t_4_0);
    runtime_util->join_thread(t_4_1);
    runtime_util->join_thread(t_4_2);
    runtime_util->join_thread(t_4_3);
    runtime_util->join_thread(t_5_0);
    runtime_util->join_thread(t_5_1);
    runtime_util->join_thread(t_5_2);
    runtime_util->join_thread(t_5_3);
    runtime_util->join_thread(t_6_0);
    runtime_util->join_thread(t_6_1);
    runtime_util->join_thread(t_6_2);
    runtime_util->join_thread(t_6_3);
    runtime_util->join_thread(t_7_0);
    runtime_util->join_thread(t_7_1);
    runtime_util->join_thread(t_7_2);
    runtime_util->join_thread(t_7_3);
}}";
    }

    public static string MakeShared()
    {
        return @"#include <tdma.h>

namespace shared {
} // namespace shared";
    }

#else
    public const string BufferType = "buffer_t";

    public const string BufferStruct = @"typedef struct buffer {
    void *vaddr;
    size_t paddr;
    uint32_t *shape;
    uint32_t *stride;
    uint32_t rank;
} buffer_t;";

    public const string MethodTable = @"typedef struct nncase_method_table {
    // float unary
    float (*float_unary_abs)(float);
    float (*float_unary_acos)(float);
    float (*float_unary_acosh)(float);
    float (*float_unary_asin)(float);
    float (*float_unary_asinh)(float);
    float (*float_unary_ceil)(float);
    float (*float_unary_cos)(float);
    float (*float_unary_cosh)(float);
    float (*float_unary_exp)(float);
    float (*float_unary_floor)(float);
    float (*float_unary_log)(float);
    float (*float_unary_logical_not)(float);
    float (*float_unary_neg)(float);
    float (*float_unary_round)(float);
    float (*float_unary_rsqrt)(float);
    float (*float_unary_sign)(float);
    float (*float_unary_sin)(float);
    float (*float_unary_sinh)(float);
    float (*float_unary_sqrt)(float);
    float (*float_unary_square)(float);
    float (*float_unary_tanh)(float);
    // float bianry
    float (*float_binary_add)(float, float);
    float (*float_binary_sub)(float, float);
    float (*float_binary_mul)(float, float);
    float (*float_binary_div)(float, float);
    float (*float_binary_min)(float, float);
    float (*float_binary_max)(float, float);
    float (*float_binary_pow)(float, float);
    float (*float_binary_logical_and)(float, float);
    float (*float_binary_mod)(float, float);
    // int32 bianry
    int32_t (*int32_binary_add)(int32_t, int32_t);
    int32_t (*int32_binary_sub)(int32_t, int32_t);
    int32_t (*int32_binary_mul)(int32_t, int32_t);
    int32_t (*int32_binary_div)(int32_t, int32_t);
    int32_t (*int32_binary_min)(int32_t, int32_t);
    int32_t (*int32_binary_max)(int32_t, int32_t);
    int32_t (*int32_binary_pow)(int32_t, int32_t);
    int32_t (*int32_binary_logical_and)(int32_t, int32_t);
    int32_t (*int32_binary_mod)(int32_t, int32_t);
    // int64 bianry
    int64_t (*int64_binary_add)(int64_t, int64_t);
    int64_t (*int64_binary_sub)(int64_t, int64_t);
    int64_t (*int64_binary_mul)(int64_t, int64_t);
    int64_t (*int64_binary_div)(int64_t, int64_t);
    int64_t (*int64_binary_min)(int64_t, int64_t);
    int64_t (*int64_binary_max)(int64_t, int64_t);
    int64_t (*int64_binary_pow)(int64_t, int64_t);
    int64_t (*int64_binary_logical_and)(int64_t, int64_t);
    int64_t (*int64_binary_mod)(int64_t, int64_t);
    // bool binary
    bool (*bool_binary_and)(bool, bool);
    bool (*bool_binary_or)(bool, bool);
    bool (*bool_binary_xor)(bool, bool);
    // multi-thread
    void *(*thread_start)(void *(*callable)(void *), void *user, size_t user_size);
    void *(*thread_end)();
} nncase_mt_t;";

    public const string Include = @"#include <stdbool.h>\n#include <stdint.h>\n#include <stddef.h>";

    public const string FixedParameters = "nncase_mt_t* nncase_mt, uint8_t* data, const uint8_t* rdata";

    public const string MainPrologue = $@"void _start(size_t func_id, uint8_t** buffers, {FixedParameters}) {{";

    public const string MainEpilogue = @"}";

    public static string Header => $@"
{Include}

{MethodTable}

{BufferStruct}
";
#endif
}
