// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR
{
    /// <summary>
    /// Tir functional Ops helper.
    /// </summary>
    public static class T
    {
        /// <summary>
        ///  Construct a vector with lanes elements
        ///   where its i-th element equals offset + i * stride.
        ///  This is useful to construct a index for a continuous vector load.
        ///  <remarks>
        ///   NOTE the stride calc by the buffer's Elemtype
        ///   if buffer's Datatype = float32*3, the 1 stride mean skip 1 float32.
        /// </remarks>
        ///  <example>
        ///  - ramp(0, 1, 3) = [0, 1, 2] = [(0 + i * 1) for i in 3]
        ///  - ramp(1, 2, 4) = [1, 3, 5, 7] = [(1 + i * 2) for i in 4]
        /// </example>
        /// </summary>
        /// <param name="offset">The base expression.</param>
        /// <param name="stride">The stride of the ramp.</param>
        /// <param name="lanes">The lanes of the expression.</param>
        public static Call Ramp(Expr offset, Expr stride, int lanes) => new Call(new TIR.Ramp(lanes), offset, stride);

        /// <summary>
        /// Load the **One** value from buffer_var.
        /// Equivalent to ((ElemType*)buffer_var)[index].
        /// <remarks>
        /// If the buffer has packed type like float32*4, but we load the index with lanes 1, so will return only one float32.
        /// </remarks>
        /// <example>
        /// case 1, type = uint32:
        ///   uint32* buffer;
        ///   auto loaded_val = buffer[index]
        /// case 2, type = float32x3
        ///   NOTE the buffer actual type is float32x3, but our index will convert it to float type.
        ///   float32x3 old_buffer;
        ///   float* buffer = static_cast<float*>(old_buffer);
        ///   NOTE then we use Ramp get the index
        ///   index = Ramp(base,1,3)
        ///   auto loaded_val = float32x3(buffer[index.v0], buffer[index.v1], buffer[index.v2]);
        ///                   = float32x3(buffer[base+(0*1)], buffer[base+(1*1)], buffer[base+(2*1)]);
        /// </example>
        /// </summary>
        /// <param name="handle">The buffer handle variable in the load expression.</param>
        /// <param name="index">The index in the load.</param>
        public static Call Load(Var handle, Expr index) => new Call(new Load(), handle, index);

        /// <summary>
        /// Store value to the buffer.
        /// Equivalent to ((DType*)buffer_var)[index] = value.
        /// where DType is the type specified by type().element_of().
        /// <example>
        /// if type = float32x3, then the store will corresponds to
        /// <code>
        ///  auto buffer = static_cast<float*>(buffer_var);
        ///  buffer[index.v0] = value.v0;
        ///  buffer[index.v1] = value.v1;
        ///  buffer[index.v2] = value.v2;
        /// </code>
        /// </example>
        /// </summary>
        /// <param name="handle">The buffer Variable.</param>
        /// <param name="value">The value we want to store.</param>
        /// <param name="index">The index in the store expression.</param>
        /// <returns></returns>
        public static Call Store(Var handle, Expr index, Expr value) => new Call(new Store(), handle, index, value);

        /// <summary>
        /// If the op is BufferLoad, it will return BufferStore
        /// If the op is Load, it will return Store.
        /// </summary>
        /// <param name="op">the op call.</param>
        /// <param name="value">update value.</param>
        /// <returns>new store call.</returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Expr Store(Expr op, Expr value) => op switch
        {
            Call load => load.Target switch
            {
                TIR.Load => T.Store((Var)load[TIR.Load.Handle], load[TIR.Load.Index], value),
                _ => throw new InvalidOperationException("Only Can build Store Op from Load!"),
            },
            TIR.BufferLoad bufload => new BufferStore(bufload.Buffer, bufload.Indices, value),
            _ => throw new InvalidOperationException("Only Can build Store Op from Load!"),
        };

        /// <summary>
        /// make a const by value and lanes.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="lanes"></param>
        /// <returns></returns>
        // public static Call MakeConst<T>(T value, Expr lanes)
        // {
        //     return new Call(new MakeConst<T>(value), lanes);
        // }

        /// <summary>
        /// get the current expr's lanes
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        // public static Call LanesOp(Expr input)
        // {
        //     return new Call(new LanesOp(), input);
        // }


        public interface ISequentialBuilder<T>
        {
            /// <summary>
            /// Add the expr items to body
            /// </summary>
            /// <param name="exprs"></param>
            /// <returns></returns>
            public T Body(params Expr[] exprs);
        }

        /// <summary>
        /// the body expr builer
        /// </summary>
        /// <typeparam name="T"></typeparam>
        public class BodyExprBuilder<T> : ISequentialBuilder<T>
          where T : BodyExpr
        {
            /// <summary>
            /// expr 
            /// </summary>
            public T Expr;

            /// <summary>
            /// ctor
            /// </summary>
            /// <param name="expr"></param>
            public BodyExprBuilder(T expr)
            {
                Expr = expr;
            }

            /// <summary>
            /// Add the expr items to body
            /// </summary>
            /// <param name="exprs"></param>
            /// <returns></returns>
            public T Body(params Expr[] exprs)
            {
                foreach (var item in exprs)
                {
                    Expr.Sequence.Add(item);
                }
                return Expr;
            }
        }

        /// <summary>
        /// get the Serial For
        /// </summary>
        /// <param name="loopVar">out index var.</param>
        /// <param name="Dom">ranges.</param>
        /// <param name="loop">loop instance.</param>
        /// <returns> the for loop </returns>
        public static BodyExprBuilder<For> Serial(out Var loopVar, Range Dom, out For loop)
        {
            loopVar = new Var(TensorType.Scalar(DataType.Int32));
            loop = new For(loopVar, Dom, LoopMode.Serial);
            return new BodyExprBuilder<For>(loop);
        }

        /// <summary>
        /// <see cref="Serial(out Var, Range, out For)"/>.
        /// </summary>
        /// <param name="loopVar"></param>
        /// <param name="Dom"></param>
        /// <returns></returns>
        public static BodyExprBuilder<For> Serial(out Var loopVar, Range Dom)
        {
            return Serial(out loopVar, Dom, out _);
        }

        /// <summary>
        /// GridWrapper for collect the for item.
        /// </summary>
        public class NestBodyExprBuilder<T> : ISequentialBuilder<T>
         where T : BodyExpr
        {
            /// <summary>
            /// contain the exprs
            /// </summary>
            public T[] Exprs;

            /// <summary>
            /// ctor
            /// <remarks>
            /// NOTE We will auto add exprs to nest list!
            /// </remarks>
            /// </summary>
            /// <param name="exprs"></param>
            public NestBodyExprBuilder(params T[] exprs)
            {
                foreach (var i in Enumerable.Range(0, exprs.Count() - 1).Reverse())
                {
                    exprs[i].Sequence.Add(exprs[i + 1]);
                }
                Exprs = exprs;
            }

            /// <summary>
            /// Wrapper Body method
            /// <see cref="For.Add(Expr[])"/>.
            /// </summary>
            /// <param name="exprs"></param>
            /// <returns></returns>
            /// <exception cref="NotImplementedException"></exception>
            public T Body(params Expr[] exprs)
            {
                foreach (var item in exprs) { Exprs.Last().Sequence.Add(item); }
                return Exprs.First();
            }
        }

        /// <summary>
        ///   for i, j in T.grid(16, 16):
        ///     with T.block():
        ///       vi, vj = T.axis.remap("SS", [i, j])
        ///       B[vi, vj] = A[vi, vj].
        /// </summary>
        /// <param name="i">outer index var.</param>
        /// <param name="j">inner index var.</param>
        /// <param name="ends">end exprs.</param>
        /// <returns>the inner for loop.</returns>
        public static NestBodyExprBuilder<For> Grid(out Var i, out Var j, (Expr i, Expr j) ends)
        {
            var builder_i = T.Serial(out i, ends.i, out var for_i);
            var builder_j = T.Serial(out j, ends.j, out var for_j);
            return new NestBodyExprBuilder<For>(for_i, for_j);
        }

        /// <summary>
        /// get grid with loops.
        /// </summary>
        /// <param name="i"></param>
        /// <param name="j"></param>
        /// <param name="ends"></param>
        /// <param name="loops"></param>
        /// <returns></returns>
        public static NestBodyExprBuilder<For> Grid(out Var i, out Var j, (Expr i, Expr j) ends, out (For i, For j) loops)
        {
            T.Serial(out i, ends.i, out loops.i);
            T.Serial(out j, ends.j, out loops.j);
            return new NestBodyExprBuilder<For>(loops.i, loops.j);
        }

        /// <summary>
        /// a named variable represents a tensor index size.
        /// </summary>
        /// <param name="name"></param>
        public static Var SizeVar(string name)
        {
            return Var.SizeVar(name);
        }

        /// <summary>
        /// create a block.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Block Block(string name)
        {
            return new Block(name);
        }

        /// <summary>
        /// Declare a new symbolic buffer.
        /// Normally buffer is created automatically during lower and build.
        /// This is only needed if user want to specify their own buffer layout.
        ///
        /// See the note below for detailed discussion on usage of buffer.
        ///  <see cref="Buffer"/>.
        /// </summary>
        /// <param name="shape">The shape of the buffer.</param>
        /// <param name="dtype">The data type of the buffer.</param>
        /// <param name="name">The name of the buffer.</param>
        /// <param name="data_handle">The data pointer in the buffer.</param>
        /// <param name="strides">The stride of the buffer.</param>
        /// <param name="elem_offset">
        ///   The beginning offset of the array to data.
        ///   In terms of number of elements of dtype.
        /// </param>
        /// <param name="scope">
        ///   The storage scope of the buffer, if not global.
        ///   If scope equals empty string, it means it is global memory.
        /// </param>
        /// <param name="data_alignment">
        ///   The alignment of data pointer in bytes.
        ///   If -1 is passed, the alignment will be set to TVM's internal default.
        /// </param>
        /// <param name="offset_factor">
        ///   The factor of elem_offset field, when set,
        ///   elem_offset is required to be multiple of offset_factor.
        ///   If 0 is pssed, the alignment will be set to 1.
        ///   if non-zero is passed, we will created a Var for elem_offset if elem_offset is not None.
        /// </param>
        /// <param name="buffer_mode">
        ///   auto_broadcast buffer allows one to implement broadcast computation
        ///   without considering whether dimension size equals to one.
        ///   TVM maps buffer[i][j][k] -> buffer[i][0][k] if dimension j's shape equals 1.
        /// </param>
        /// <returns>Buffer.</returns>
        public static Buffer DeclBuffer(IR.Tuple shape, DataType? dtype = null, string name = "buffer", Var? data_handle = null, IR.Tuple? strides = null, Expr? elem_offset = null, string scope = "", int data_alignment = -1, int offset_factor = 0, BufferMode buffer_mode = BufferMode.Default)
        {
            dtype ??= DataType.Float32;

            if (offset_factor != 0 && elem_offset is null)
            {
                elem_offset = Var.Scalar($"{name}_elem_offset", shape[0].CheckedDataType);
            }

            if (data_handle is null)
            {
                data_handle = Var.Handle(name, dtype, scope);
            }

            elem_offset ??= (Const)0;
            if (data_alignment <= 0)
            {
                data_alignment = 128; // TODO add useage.
            }

            if (offset_factor == 0)
            {
                offset_factor = 1;
            }

            // compute the default stride.
            Expr acc = 1;
            Expr prod(Expr dim) { acc = dim * acc; return acc; }
            strides ??= new(new Expr[] { 1 }.Concat(
              Enumerable.Range(0, shape.Count - 1).Reverse().Select(i => prod(shape[i + 1]))
            ).Reverse());

            return new Buffer(shape, name, data_handle, strides, elem_offset, scope, data_alignment, offset_factor, buffer_mode);
        }

        /// <summary>
        /// script function builder
        /// </summary>
        public class FunctionBuilder
        {
            readonly Function func;
            /// <summary>
            /// cotr
            /// </summary>
            /// <param name="func"></param>
            public FunctionBuilder(Function func)
            {
                this.func = func;
            }

            /// <summary>
            /// add the body items
            /// </summary>
            /// <param name="exprs">the expr instance.</param>
            /// <returns> the func instance.</returns>
            public Function Body(params Expr[] exprs)
            {
                var body = (Sequential)func.Body;
                foreach (var item in exprs)
                {
                    body.Add(item);
                }

                return func;
            }
        }

        /// <summary>
        /// The script for build funciont with Sequential body.
        /// <code>
        ///  var func = T.PrimFunc("func", A.Handle, n, m).Add(
        ///  T.Serial(out var i, n, out var fi).Add(
        ///  T.Serial(out var j, m, out var fj).Add(
        ///    T.Block("init").
        ///    Remap(out var vi, out var vj, (fi, fj), "SS").
        ///    Init(T.Store(A[vi, vj], 1)).Add(
        ///      T.Store(A[vi, vj], vi + vj)
        ///    )
        ///  )
        /// ));
        /// </code>
        /// </summary>
        /// <param name="name"></param>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static FunctionBuilder PrimFunc(string name, params Expr[] parameters)
        {
            return new(new Function(name, new Sequential(), parameters));
        }

        /// <summary>
        /// create the handle var.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        public static Var Handle(string name, DataType dtype)
        {
            return Var.Handle(name, dtype);
        }

        public class IfThenElseBuilder
        {
            readonly Sequential ThenBranch;
            readonly Sequential ElseBranch;
            readonly Expr Condition;
            public IfThenElseBuilder(Expr condition)
            {
                ThenBranch = new();
                ElseBranch = new();
                Condition = condition;
            }

            public IfThenElseBuilder Then(params Expr[] exprs)
            {
                foreach (var item in exprs)
                {
                    ThenBranch.Add(item);
                }

                return this;
            }

            public IfThenElseBuilder Else(params Expr[] exprs)
            {
                foreach (var item in exprs)
                {
                    ElseBranch.Add(item);
                }

                return this;
            }

            public IfThenElse ToExpr()
            {
                return new IfThenElse(Condition, ThenBranch, ElseBranch);
            }

            public static implicit operator Expr(IfThenElseBuilder builder)
            {
                return builder.ToExpr();
            }
        }

        /// <summary>
        /// rethen the IfThenElseBuilder.
        /// </summary>
        /// <param name="condition"></param>
        /// <returns></returns>
        public static IfThenElseBuilder If(Expr condition)
        {
            return new(condition);
        }
    }
}