// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.Builtin
{
    [System.AttributeUsage(System.AttributeTargets.All, Inherited = false, AllowMultiple = true)]
    public sealed class CallEffectAttribute : System.Attribute
    {
        private CallEffectMode _mode;

        public CallEffectMode Mode => _mode;

        public CallEffectAttribute(CallEffectMode mode)
        {
            _mode = mode;
        }
    }

    /// <summary>
    /// Reinterpret the value using the target type.
    /// </summary>
    public sealed record Reinterpret() : Op
    {
    }

    /// <summary>
    ///  Marks a condition is likely going to happen.
    /// </summary>
    public sealed record likely() : Op { }

    /// <summary>
    ///  Bitwise and operator.
    /// </summary>
    public sealed record bitwise_and() : Op { }

    /// <summary>
    ///  Bitwise or operator.
    /// </summary>
    public sealed record bitwise_or() : Op { }

    /// <summary>
    ///  Bitwise xor operator.
    /// </summary>
    public sealed record bitwise_xor() : Op { }

    /// <summary>
    ///  Bitwise not operator.
    /// </summary>
    public sealed record bitwise_not() : Op { }

    /// <summary>
    ///  Left shift.
    /// </summary>
    public sealed record shift_left() : Op { }

    /// <summary>
    ///  Right shift.
    /// </summary>
    public sealed record shift_right() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  Construct a big uint that may not be representable by int64
    ///
    ///  Expr large_uint_imm(uint32_t v0, uin32_t v1) {
    ///    return (v1. << 32) | v0;
    ///  }
    /// </summary>
    public sealed record large_uint_imm() : Op { }

    /// <summary>
    ///  Execute a multiplication between two Q-numbers x and y
    /// followed by a right shift s
    /// The default rounding rule is to the nearest value, rounding half up
    /// (i.e., round(x.1) = x and round (x.5) = x+1).
    /// </summary>
    public sealed record q_multiply_shift() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  Handle address_of(Load ///op) {
    ///     return &op->buffer_var[index];
    ///  }.
    /// </summary>
    public sealed record address_of() : Op { }

    /// <summary>
    ///  Same as select, used for unsafe memory access.
    ///
    ///  Type if_then_else(cond, a, b) {
    ///    return cond ? a : b;
    ///  }.
    /// </summary>
    public sealed record if_then_else() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  bool isnullptr(void/// handle) {
    ///     return handle == nullptr
    ///  }.
    /// </summary>
    public sealed record isnullptr() : Op { }

    /// <summary>
    ///  Check if value is nan.
    /// </summary>
    public sealed record isnan() : Op { }

    /// <summary>
    ///  Popcount.
    /// </summary>
    public sealed record popcount() : Op { }

    /// <summary>
    ///  Fused multiply add
    ///
    ///  Type fma(a, b, c) {
    ///    return a // b + c;
    ///  }.
    /// </summary>
    public sealed record fma() : Op { }

    /// <summary>
    ///  Call an extern C function with given name
    ///        and signature from the types of args in the runtime environment.
    ///
    ///  Type call_extern(name, args...) {
    ///     return dlsym(name)(args...);
    ///  }
    ///
    /// \note This intrinsic does not provide any type checking,
    ///       and is main used for backward compatibility reasons.
    ///       Always consider use pre-registered and typed tvm::Op first.
    /// </summary>
    public sealed record call_extern() : Op { }

    /// <summary>
    ///  Call an pure extern C function with given name
    ///        and signature from the types of args in the runtime environment.
    ///
    ///  Type call_pure_extern(name, args...) {
    ///     return dlsym(name)(args...);
    ///  }
    ///
    /// \note This intrinsic does not provide any type checking,
    ///       and is main used for backward compatibility reasons.
    ///       Always consider use pre-registered and typed tvm::Op first.
    /// </summary>
    public sealed record call_pure_extern() : Op { }

    /// <summary>
    ///  Call an LLVM intrinsic with a given intrinsic id
    ///        and signature from the types of args in the runtime environment.
    ///
    ///  Type call_llvm_pure_intrin(intrin_id, args...) {
    ///     return dlsym(name)(args...);
    ///  }
    ///
    /// \note This op does not provide any type checking.
    /// </summary>
    public sealed record call_llvm_intrin() : Op { }

    /// <summary>
    ///  Call an LLVM pure intrinsic with a given intrinsic id
    ///        and signature from the types of args in the runtime environment.
    ///
    ///  Type call_llvm_pure_intrin(intrin_id, args...) {
    ///     return dlsym(name)(args...);
    ///  }
    ///
    /// \note This op does not provide any type checking.
    /// </summary>
    public sealed record call_llvm_pure_intrin() : Op { }

    /// <summary>
    ///  Call an SPIRV pure GLSL450 intrinsic.
    ///
    ///  Type call_spirv_pure_glsl450(intrin_id, args...) {
    ///     return dlsym(name)(args...);
    ///  }
    ///
    /// \note This op does not provide any type checking.
    /// </summary>
    public sealed record call_spirv_pure_glsl450() : Op { }

    // TODO(tvm-team) revisit the builtins below
    // some of them can simply become ops with special codegen attr.
    /// <summary>
    ///  Prefetch a cacheline.
    /// </summary>
    public sealed record prefetch() : Op { }

    /// <summary>
    ///  Get head access address with memory access pattern info.
    ///
    ///  This operator also marks range of the memory access
    ///  The offset and extent are in unit of the DType(including vectorization factor).
    ///  rw_mask is a bit_mask setting whether the access is a read(1) or write(2).
    ///  The access is assume to happen in the current expression.
    /// <example>
    /// <code>
    ///  PtrType access_ptr(Handle data, int offset, int extent) {
    ///     DType == dtype.type();
    ///    return &data[offset];
    ///  }
    /// </code>
    /// </example>
    /// </summary>
    /// <param name="DType"> the read data type.</param>
    /// <param name="AccessMode"> access mode. </param>
    public sealed record AccessPtr(DataType DType, AccessMode AccessMode) : Op
    {
        public static readonly ParameterInfo Data = new(typeof(AccessPtr), 0, "data");
        public static readonly ParameterInfo Offset = new(typeof(AccessPtr), 1, "offset");
        public static readonly ParameterInfo Extent = new(typeof(AccessPtr), 2, "extent");
    }

    /// <summary>
    ///  Create a function local static handle that iniitalizes to nullptr.
    ///  can be used to cache function local static resources.
    /// </summary>
    public sealed record static_handle() : Op { }

    /// <summary>
    ///  Return a unique context id, used for hint of workspace separation.
    ///  Different context id ganrantees not having overlapping workspace.
    /// </summary>
    public sealed record context_id() : Op { }

    /// <summary>
    ///  tuple is not an actual function and cannot codegen.
    ///  It is used to represent tuple structure in value field of AttrStmt,
    ///  for the sake of giving hint to optimization.
    ///
    ///  Handle tuple(value0, value1, ..., value_n).
    /// </summary>
    public sealed record tuple() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  Type struct_get(StructType/// arr, int index, int field_id) {
    ///     return arr[index]->field;
    ///  }
    /// \sa TVMStructFieldKind.
    /// </summary>
    public sealed record struct_get() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  Handle struct_set(StructType/// arr, int index, int field_id, value) {
    ///     arr[index]->field = value;
    ///  }
    /// \sa TVMStructFieldKind.
    /// </summary>
    public sealed record struct_set() : Op { }

    /// <summary>
    ///  See pseudo code
    /// Type lookup_param(String param_name) {
    ///     return __param__param_name;
    /// }.
    /// </summary>
    public sealed record lookup_param() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  void throw_last_error() {
    ///    throw TVMGetLastError();
    ///  }.
    /// </summary>
    public sealed record throw_last_error() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  dtype in {shape, array, arg_value, arg_tcode}
    ///
    ///  Handle stack_alloca(string dtype, int num) {
    ///     return new on stack dtype[num];
    ///  }.
    /// </summary>
    public sealed record stack_alloca() : Op { }

    /// <summary>
    ///  Allocate a shape tuple on stack, return the handle.
    ///
    ///  Handle stack_make_shape(list args) {
    ///     ret = alloca stack int64_t[len(args)];
    ///     for i in range(len(args)):
    ///        ret[i] = args[i]
    ///     return &ret[0];
    ///  }.
    /// </summary>
    public sealed record stack_make_shape() : Op { }

    /// <summary>
    ///  Allocate a NDArray(DLTensor) on stack, return the handle.
    ///
    ///  Type stack_make_array(Expr data,
    ///                            Expr shape,
    ///                            Expr strides,
    ///                            Expr ndim,
    ///                            Expr dtype,
    ///                            Expr elem_offset) {
    ///     ret = alloca stack DLTensor();
    ///     ret->data = data;
    ///     ret->shape = shape;
    ///     ret->strides = strides != 0 ? strides : nullptr;
    ///     ret->ndim = ndim;
    ///     ret->dtype = dtype.type();
    ///     ret->byte_offset = elem_offset /// sizeof(dtype);
    ///     return ret;
    ///  }.
    /// </summary>
    public sealed record stack_make_array() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  return_type call_packed(name, TVMValue/// args) {
    ///     TVMValue ret_value;
    ///     int ret_code;
    ///     ModuleNode/// env = GetCurrentEnv();
    ///     const PackedFunc/// f = env->GetFuncFromEnv(name);
    ///     (///f)(args, type_code_of(args), len(args), &ret_value, &ret_code);
    ///     // return type can be int, float, handle.
    ///     return cast(return_type, ret_value.v_return_type);
    ///  }.
    /// </summary>
    public sealed record call_packed() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    /// return_type call_packed(fname, TVMValue/// args) {
    /// 	   int ret_code;
    ///     TVMValue ret_value;
    ///     (///fname)(args, type_code_of(args), len(args), &ret_value, &ret_code);
    ///     return cast(return_type, ret_value.v_return_type);
    ///  }.
    /// </summary>
    public sealed record call_cpacked() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  return_type call_trace_packed(name, TVMValue/// args) {
    ///     ModuleNode/// env = GetCurrentEnv();
    ///     const PackedFunc/// f = env->GetFuncFromEnv(name);
    ///     (///f)(args, type_code_of(args), len(args));
    ///     // return type can be int, float, handle.
    ///     return cast(return_type, ret_value.v_return_type);
    ///  }.
    /// </summary>
    public sealed record call_trace_packed() : Op { }

    /// <summary>
    ///  See pesudo code
    ///  Mark the content as thread local context, can get optimized
    ///  by only call the call once at thread start.
    ///
    ///  Do not allow nesting(getting a thread context from another).
    ///
    ///  Handle thread_context(Expr call) {
    ///     return call;
    ///  }.
    /// </summary>
    public sealed record thread_context() : Op { }

    /// <summary>
    ///  Lowered version of call packed, the space of value and
    ///  type codes are explicitly allocated.
    ///
    ///  return_type call_packed_lowered(name,
    ///                                      TVMValue/// value_stack,
    ///                                      int/// tcode_stack,
    ///                                      int begin,
    ///                                      int end) {
    ///     ModuleNode/// env = GetCurrentEnv();
    ///     const PackedFunc/// f = env->GetFuncFromEnv(name);
    ///     f->CallPacked(TVMArgs(value_stack[begin:end],
    ///                           tcode_stack[begin:end]),
    ///                   TVMRetValue(value_stack + end, tcode_stack + end));
    ///     // return type can be int, float, handle.
    ///     return cast(return_type, load_return_from(tcode_stack + end))
    ///  }.
    /// </summary>
    public sealed record call_packed_lowered() : Op { }

    /// <summary>
    ///  Lowered version of call c-packed, the space of value and
    ///  type codes are explicitly allocated.
    ///
    ///  int call_packed_lowered(fname,
    ///                              TVMValue/// value_stack,
    ///                              int/// tcode_stack,
    ///                              int begin,
    ///                              int end) {
    ///     fname(TVMArgs(value_stack[begin:end], tcode_stack[begin:end]),
    ///                   TVMRetValue(value_stack + end, tcode_stack + end));
    ///  }.
    /// </summary>
    public sealed record call_cpacked_lowered() : Op { }

    /// <summary>
    ///  Lowered version of trace intrinsic, the space of value and
    ///  type codes are explicitly allocated. The return value is the
    ///  (end - 1) value on the stack.
    ///
    ///  return_type call_trace_packed_lowered(name,
    ///                                            TVMValue/// value_stack,
    ///                                            int/// tcode_stack,
    ///                                            int begin,
    ///                                            int end) {
    ///     ModuleNode/// env = GetCurrentEnv();
    ///     const PackedFunc/// f = env->GetFuncFromEnv(name);
    ///     f->CallPacked(TVMArgs(value_stack[begin:end],
    ///                           tcode_stack[begin:end]),
    ///                   TVMRetValue(value_stack + end, tcode_stack + end));
    ///     // return type can be int, float, handle.
    ///     return cast(return_type, load_return_from(tcode_stack + end))
    ///  }.
    /// </summary>
    public sealed record call_trace_packed_lowered() : Op { }

    /// <summary>
    ///  See pseudo code
    ///
    ///  int storage_sync(std::string storage_scope) {
    ///     __sync(storage_scope);
    ///     return 0;
    ///  }.
    /// </summary>
    public sealed record storage_sync() : Op { }

    /// <summary>
    ///  See pseudo code
    ///
    ///  Type warp_shuffle(mask, Type value, warp_id, width, warp_size) {
    ///    return (value passed in by warp indicated by this_warp_id);
    ///  }
    ///
    ///  Type warp_shuffle_up(mask, Type value, offset, width, warp_size) {
    ///    return (value passed in by warp indicated by this_warp_id - offset);
    ///  }
    ///
    ///  Type warp_shuffle_down(mask, Type value, offset, width, warp_size) {
    ///    return (value passed in by warp indicated by this_warp_id + offset);
    ///  }
    ///
    ///  unsigned warp_activemask() {
    ///    return (32-bit mask of currently active threads in the calling warp);
    ///  }
    ///
    ///  Parameter warp_id indicates the source thread ID in a warp.
    ///
    ///  Parameter offset indicates the relative distance to this_warp_id.
    ///
    ///  Parameter width indicates the number of threads involved in one
    ///  shuffle. See CUDA document for __shfl_sync, __shfl_up_sync,
    ///  __shfl_down_sync and __activemask.
    ///
    ///  Parameter warp_size is the size of a warp, which helps a backend
    ///  to determine wheter the width paramter is legal.
    ///
    /// </summary>
    public sealed record warp_shuffle() : Op { }
    public sealed record warp_shuffle_up() : Op { }
    public sealed record warp_shuffle_down() : Op { }
    public sealed record warp_activemask() : Op { }

    /// <summary>
    ///  Initialize the global barrier.
    ///  Call this at beginning of kernel that need global barrier.
    /// </summary>
    public sealed record global_barrier_kinit() : Op { }

    /// <summary>
    ///  See pesudo code
    ///
    ///  void thread_allreduce(UIntImm size, Expr source0, ..., Expr cond,
    ///                            Var reduce_temp0, .., Var thread_idx1, ...) {
    ///     // constraint by the other thread_idx remain the same.
    ///     // reduce_temp is used to save intermediate result.
    ///     reduce_temp0, ... = reduce(combiner, source0, ..., cond
    ///       over [thread_idx1, thread_idx2] passed by any caller)
    ///  }.
    /// </summary>
    public sealed record thread_allreduce() : Op { }

    // TODO(tvm-team) TensorCore specific intrinsics should be directly registered under
    //                cuda. namespace and used through op.
    /// <summary>
    ///  tvm intrinsic for tensor core load operators.
    ///
    ///  void load_matrix_sync(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
    ///                            Expr index, Expr buffer_ptr, Expr stride,
    ///                            StringImm layout) {
    ///    // m, n, k are the shape of wmma fragment.
    ///    // Determine fragment layout(column-major or row major) by layout.
    ///    // fragments must be in 'wmma.matrix_a' or 'wmma.matrix_b' scope.
    ///    nvcuda::wmma::load_matrix_sync(fragment[index], buffer_ptr, stride);
    ///  }.
    /// </summary>
    public sealed record load_matrix_sync() : Op { }

    /// <summary>
    ///  tvm intrinsic for tensor core mma_sync operators.
    ///
    ///  void mma_sync(Var fragment_d, Expr index_d,
    ///                    Var fragment_a, Expr index_a,
    ///                    Var fragment_b, Expr index_b,
    ///                    Var fragment_c, Expr index_c) {
    ///    nvcuda::wmma::mma_sync(fragment_d[index_d], fragment_a[index_a],
    ///                           fragment_b[index_b], fragment_c[index_c]);
    ///  }.
    /// </summary>
    public sealed record mma_sync() : Op { }

    /// <summary>
    ///  tvm intrinsic for tensor core bmma_sync operators.
    ///
    ///  void bmma_sync(Var fragment_d, Expr index_d,
    ///                     Var fragment_a, Expr index_a,
    ///                     Var fragment_b, Expr index_b,
    ///                     Var fragment_c, Expr index_c) {
    ///    nvcuda::wmma::bmma_sync(fragment_d[index_d], fragment_a[index_a],
    ///                           fragment_b[index_b], fragment_c[index_c]);
    ///  }.
    /// </summary>
    public sealed record bmma_sync() : Op { }

    /// <summary>
    ///  tvm intrinsic for tensor core fill_fragment operators.
    ///
    ///  void fill_fragment(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
    ///                         Expr index, Expr value) {
    ///    // m, n, k are the shape of wmma fragment
    ///    // fragments must be in 'wmma.accumulator' scope.
    ///    nvcuda::wmma::fill_fragment(fragment[index], value);
    ///  }.
    /// </summary>
    public sealed record fill_fragment() : Op { }

    /// <summary>
    ///  tvm intrinsic for tensor core store operators.
    ///
    ///  void store_matrix_sync(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
    ///                             Expr index, Expr buffer_ptr, Expr stride,
    ///                             StringImm layout) {
    ///    // m, n, k are the shape of wmma fragment
    ///    // fragments must be in 'wmma.accumulator' scope.
    ///    nvcuda::wmma::store_matrix_sync(fragment[index], buffer_ptr, stride, layout);
    ///  }.
    /// </summary>
    public sealed record store_matrix_sync() : Op { }

    // TODO(tvm-team) replace the usage of the vector operations by Shuffle.
    /// <summary>
    ///  Get the high level half of the vector.
    /// </summary>
    public sealed record vectorhigh() : Op { }

    /// <summary>
    ///  Get the low-level half of the vector.
    /// </summary>
    public sealed record vectorlow() : Op { }

    /// <summary>
    ///  Concat two vectors.
    /// </summary>
    public sealed record vectorcombine() : Op { }

    /// <summary>
    ///  atomic add instruction, corresponding e.g. to atomicAdd in CUDA.
    /// </summary>
    public sealed record atomic_add() : Op { }
    /// <summary>
    ///  Create a texture 2d memory allocation.
    /// </summary>
    public sealed record texture2d_alloca() : Op { }

    /// <summary>
    ///  Store to texture 2d memory.
    /// </summary>
    public sealed record texture2d_store() : Op { }

    /// <summary>
    ///  Load from texture 2d memory.
    /// </summary>
    public sealed record texture2d_load() : Op { }

    /// <summary>  The kind of structure field info used in intrinsic ///. </summary>
    enum TVMStructFieldKind : int
    {
        // array head address
        kArrAddr,
        kArrData,
        kArrShape,
        kArrStrides,
        kArrNDim,
        kArrTypeCode,
        kArrTypeBits,
        kArrTypeLanes,
        kArrByteOffset,
        kArrDeviceId,
        kArrDeviceType,
        kArrKindBound_,

        // TVMValue field
        kTVMValueContent,
        kTVMValueKindBound_,
    }

;
}