/* Copyright 2020 Canaan Inc.
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
#include "evaluate_stack.h"
#include <nncase/runtime/stackvm/op_reader.h>
#include <nncase/runtime/stackvm/runtime_module.h>
#include <nncase/runtime/stackvm/kernel_context.h>

BEGIN_NS_NNCASE_RT_STACKVM

class stackvm_runtime_module : public runtime_module, private op_visitor
{
public:
    static NNCASE_INLINE_VAR constexpr size_t MAX_GENERAL_REGS = 32;

protected:
    result<void> initialize_core(runtime_module_init_context &context) noexcept override;
    result<runtime_tensor> allocate_input_tensor(size_t index) noexcept override;
    result<runtime_tensor> allocate_output_tensor(size_t index) noexcept override;
    result<void> validate_input_tensor(size_t index, runtime_tensor tensor) noexcept override;
    result<void> validate_output_tensor(size_t index, runtime_tensor tensor) noexcept override;
    result<void> run_core() noexcept override;

    using op_visitor::visit;
    result<void> visit(const nop_op_t &op) noexcept override;
    result<void> visit(const br_op_t &op) noexcept override;
    result<void> visit(const br_true_op_t &op) noexcept override;
    result<void> visit(const br_false_op_t &op) noexcept override;
    result<void> visit(const ret_op_t &op) noexcept override;
    result<void> visit(const call_op_t &op) noexcept override;
    result<void> visit(const ecall_op_t &op) noexcept override;
    result<void> visit(const throw_op_t &op) noexcept override;
    result<void> visit(const break_op_t &op) noexcept override;

    result<void> visit(const ldc_i4_op_t &op) noexcept override;
    result<void> visit(const ldnull_op_t &op) noexcept override;
    result<void> visit(const ldc_i4_0_op_t &op) noexcept override;
    result<void> visit(const ldc_i4_1_op_t &op) noexcept override;
    result<void> visit(const ldc_r4_op_t &op) noexcept override;
    result<void> visit(const ldind_i1_op_t &op) noexcept override;
    result<void> visit(const ldind_i2_op_t &op) noexcept override;
    result<void> visit(const ldind_i4_op_t &op) noexcept override;
    result<void> visit(const ldind_i_op_t &op) noexcept override;
    result<void> visit(const ldind_u1_op_t &op) noexcept override;
    result<void> visit(const ldind_u2_op_t &op) noexcept override;
    result<void> visit(const ldind_u4_op_t &op) noexcept override;
    result<void> visit(const ldind_u_op_t &op) noexcept override;
    result<void> visit(const ldind_br2_op_t &op) noexcept override;
    result<void> visit(const ldind_r4_op_t &op) noexcept override;
    result<void> visit(const stind_i1_op_t &op) noexcept override;
    result<void> visit(const stind_i2_op_t &op) noexcept override;
    result<void> visit(const stind_i4_op_t &op) noexcept override;
    result<void> visit(const stind_i_op_t &op) noexcept override;
    result<void> visit(const stind_br2_op_t &op) noexcept override;
    result<void> visit(const stind_r4_op_t &op) noexcept override;

    result<void> visit(const lea_gp_op_t &op) noexcept override;
    result<void> visit(const lea_buffer_op_t &op) noexcept override;
    result<void> visit(const ldelem_i1_op_t &op) noexcept override;
    result<void> visit(const ldelem_i2_op_t &op) noexcept override;
    result<void> visit(const ldelem_i4_op_t &op) noexcept override;
    result<void> visit(const ldelem_i_op_t &op) noexcept override;
    result<void> visit(const ldelem_u1_op_t &op) noexcept override;
    result<void> visit(const ldelem_u2_op_t &op) noexcept override;
    result<void> visit(const ldelem_u4_op_t &op) noexcept override;
    result<void> visit(const ldelem_u_op_t &op) noexcept override;
    result<void> visit(const ldelem_br2_op_t &op) noexcept override;
    result<void> visit(const ldelem_r4_op_t &op) noexcept override;
    result<void> visit(const stelem_i1_op_t &op) noexcept override;
    result<void> visit(const stelem_i2_op_t &op) noexcept override;
    result<void> visit(const stelem_i4_op_t &op) noexcept override;
    result<void> visit(const stelem_i_op_t &op) noexcept override;
    result<void> visit(const stelem_br2_op_t &op) noexcept override;
    result<void> visit(const stelem_r4_op_t &op) noexcept override;
    result<void> visit(const ldarg_op_t &op) noexcept override;
    result<void> visit(const ldarg_0_op_t &op) noexcept override;
    result<void> visit(const ldarg_1_op_t &op) noexcept override;
    result<void> visit(const ldarg_2_op_t &op) noexcept override;
    result<void> visit(const ldarg_3_op_t &op) noexcept override;
    result<void> visit(const ldarg_4_op_t &op) noexcept override;
    result<void> visit(const ldarg_5_op_t &op) noexcept override;
    result<void> visit(const stshape_op_t &op) noexcept override;
    result<void> visit(const stpaddings_op_t &op) noexcept override;

    result<void> visit(const dup_op_t &op) noexcept override;
    result<void> visit(const pop_op_t &op) noexcept override;

    result<void> visit(const neg_op_t &op) noexcept override;
    result<void> visit(const add_op_t &op) noexcept override;
    result<void> visit(const sub_op_t &op) noexcept override;
    result<void> visit(const mul_op_t &op) noexcept override;
    result<void> visit(const div_op_t &op) noexcept override;
    result<void> visit(const div_u_op_t &op) noexcept override;
    result<void> visit(const rem_op_t &op) noexcept override;
    result<void> visit(const rem_u_op_t &op) noexcept override;
    result<void> visit(const and_op_t &op) noexcept override;
    result<void> visit(const or_op_t &op) noexcept override;
    result<void> visit(const xor_op_t &op) noexcept override;
    result<void> visit(const not_op_t &op) noexcept override;
    result<void> visit(const shl_op_t &op) noexcept override;
    result<void> visit(const shr_op_t &op) noexcept override;
    result<void> visit(const shr_u_op_t &op) noexcept override;
    result<void> visit(const clt_op_t &op) noexcept override;
    result<void> visit(const clt_u_op_t &op) noexcept override;
    result<void> visit(const cle_op_t &op) noexcept override;
    result<void> visit(const cle_u_op_t &op) noexcept override;
    result<void> visit(const ceq_op_t &op) noexcept override;
    result<void> visit(const cge_op_t &op) noexcept override;
    result<void> visit(const cge_u_op_t &op) noexcept override;
    result<void> visit(const cgt_op_t &op) noexcept override;
    result<void> visit(const cgt_u_op_t &op) noexcept override;
    result<void> visit(const cne_op_t &op) noexcept override;

    result<void> visit(const conv_i1_op_t &op) noexcept override;
    result<void> visit(const conv_i2_op_t &op) noexcept override;
    result<void> visit(const conv_i4_op_t &op) noexcept override;
    result<void> visit(const conv_i_op_t &op) noexcept override;
    result<void> visit(const conv_u1_op_t &op) noexcept override;
    result<void> visit(const conv_u2_op_t &op) noexcept override;
    result<void> visit(const conv_u4_op_t &op) noexcept override;
    result<void> visit(const conv_u_op_t &op) noexcept override;
    result<void> visit(const conv_br2_op_t &op) noexcept override;
    result<void> visit(const conv_r4_op_t &op) noexcept override;

    result<void> visit(const tensor_batch_to_space_op_t &op) noexcept override;
    result<void> visit(const tensor_binary_op_t &op) noexcept override;
    result<void> visit(const tensor_broadcast_op_t &op) noexcept override;
    result<void> visit(const tensor_call_op_t &op) noexcept override;
    result<void> visit(const tensor_concat_op_t &op) noexcept override;
    result<void> visit(const tensor_conv2d_op_t &op) noexcept override;
    result<void> visit(const tensor_convert_op_t &op) noexcept override;
    result<void> visit(const tensor_dequantize_op_t &op) noexcept override;
    result<void> visit(const tensor_lut1d_op_t &op) noexcept override;
    result<void> visit(const tensor_pad_op_t &op) noexcept override;
    result<void> visit(const tensor_quantize_op_t &op) noexcept override;
    result<void> visit(const tensor_reduce_op_t &op) noexcept override;
    result<void> visit(const tensor_reduce_window2d_op_t &op) noexcept override;
    result<void> visit(const tensor_slice_op_t &op) noexcept override;
    result<void> visit(const tensor_transpose_op_t &op) noexcept override;
    result<void> visit(const tensor_unary_op_t &op) noexcept override;

private:
    uintptr_t pc() const noexcept;
    result<void> pc(uintptr_t value) noexcept;
    result<void> pc_relative(intptr_t offset) noexcept;
    result<padding> pop_padding() noexcept;
    result<uintptr_t> pop_addr() noexcept;
    runtime_axis_t as_runtime_axis(const runtime_shape_t &shape);
    result<scalar> pop_scalar(datatype_t type) noexcept;
    kernels::kernel_context &kernel_context() noexcept;

    template <class T>
    result<T> pop_addr() noexcept
    {
        try_var(addr, pop_addr());
        return reinterpret_cast<T>(addr);
    }

private:
    std::unique_ptr<gsl::byte[]> data_;
    gsl::span<const gsl::byte> rdata_;
    gsl::span<const gsl::byte> text_;
    evaluate_stack stack_;
    size_t call_depth_;
    std::array<uintptr_t, MAX_GENERAL_REGS> regs_;
    std::vector<runtime_shape_t> shape_regs_;
    std::vector<runtime_paddings_t> paddings_regs_;
    stackvm_kernel_context kernel_context_;
};

END_NS_NNCASE_RT_STACKVM
