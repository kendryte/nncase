// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
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
        private static readonly Dictionary<char, int> globalLoopVarIndex = new()
        {
            { 'i', 0 },
            { 'j', 0 },
            { 'k', 0 },
            { 'l', 0 },
        };

        private static readonly Dictionary<string, int> globalSizeVarIndex = new()
        {
        };

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
        /// Equivalent to ((ElemType*)buffer_var)[index]
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
        public static Call Store(Var handle, Expr value, Expr index) => new Call(new Store(), handle, value, index);


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


        static Var GetUniqueLoopVar()
        {
            KeyValuePair<char, int> func(KeyValuePair<char, int> l, KeyValuePair<char, int> r)
            {
                if (l.Value == r.Value)
                {
                    return l.Key < r.Key ? l : r;
                }
                return l.Value < r.Value ? l : r;
            }
            var name = globalLoopVarIndex.Aggregate(func).Key;
            return GetUniqueLoopVar(name);
        }

        static Var GetUniqueLoopVar(char name)
        {
            int count = globalLoopVarIndex[name];
            globalLoopVarIndex[name]++;
            return new Var($"{name}_{count}", TensorType.Scalar(DataType.Int32));
        }

        /// <summary>
        /// get the Serial For
        /// </summary>
        /// <param name="loop_var">out index var.</param>
        /// <param name="begin">begin expr.</param>
        /// <param name="end">end expr.</param>
        /// <returns> the for loop </returns>
        public static For Serial(out Var loop_var, Expr begin, Expr end)
        {
            loop_var = GetUniqueLoopVar();
            return new For(loop_var, begin, end, ForMode.Serial);
        }

        /// <summary>
        /// GridWrapper for collect the for item.
        /// </summary>
        public class GridWrapper
        {
            public For[] ForList;
            public GridWrapper(params For[] for_list)
            {
                ForList = for_list;
            }
            /// <summary>
            /// Wrapper Body method
            /// <see cref="For.Body(Expr[])"/>
            /// </summary>
            /// <param name="exprs"></param>
            /// <returns> the outter for loop instance. </returns>
            public For Body(params Expr[] exprs)
            {
                ForList.Last().Body(exprs);
                return ForList.First();
            }
        }

        /// <summary>
        ///   for i, j in T.grid(16, 16):
        ///     with T.block():
        ///       vi, vj = T.axis.remap("SS", [i, j])
        ///       B[vi, vj] = A[vi, vj]
        /// </summary>
        /// <param name="i">outer index var.</param>
        /// <param name="j">inner index var.</param>
        /// <param name="ends">end exprs.</param>
        /// <returns>the inner for loop.</returns>
        public static GridWrapper Grid(out Var i, out Var j, (Expr i, Expr j) ends)
        {
            i = GetUniqueLoopVar('i');
            j = GetUniqueLoopVar('j');
            var for_i = new For(i, 0, ends.i, ForMode.Serial);
            var for_j = new For(j, 0, ends.j, ForMode.Serial);
            for_i.Body(for_j);
            return new GridWrapper(for_i, for_j);
        }

        /// <summary>
        /// a named variable represents a tensor index size
        /// </summary>
        /// <param name="Name"></param>
        /// <param name="DType"></param>        
        public static SizeVar SizeVar(string Name, ElemType DType = ElemType.Int32)
        {
            string newName = Name;
            if (!globalSizeVarIndex.TryGetValue(Name, out var i))
            {
                i = 0;
                globalSizeVarIndex.Add(Name, i);
                return new SizeVar(Name, new DataType(DType, 1));
            }
            globalSizeVarIndex[Name]++;
            return new SizeVar($"{Name}_{i}", new DataType(DType, 1));
        }
    }
}