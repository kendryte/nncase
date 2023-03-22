﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using Nncase.IR;
using Nncase.TIR.Builders;

namespace Nncase.TIR;

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
    ///   float* buffer = static_cast&lt;float*&gt;(old_buffer);
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
    /// get the nop op.
    /// </summary>
    public static Call Nop() => new Call(new Nop());

    /// <summary>
    /// Store value to the buffer.
    /// Equivalent to ((DType*)buffer_var)[index] = value.
    /// where DType is the type specified by type().element_of().
    /// <example>
    /// if type = float32x3, then the store will corresponds to
    /// <code>
    ///  auto buffer = static_cast{float*}(buffer_var);
    ///  buffer[index.v0] = value.v0;
    ///  buffer[index.v1] = value.v1;
    ///  buffer[index.v2] = value.v2;
    /// </code>
    /// </example>
    /// </summary>
    /// <param name="handle">The buffer Variable.</param>
    /// <param name="index">The index in the store expression.</param>
    /// <param name="value">The value we want to store.</param>
    public static Call Store(Var handle, Expr index, Expr value) => new Call(new Store(), handle, index, value);

    /// <summary>
    /// If the op is BufferLoad, it will return BufferStore
    /// If the op is Load, it will return Store.
    /// </summary>
    /// <param name="op">the op call.</param>
    /// <param name="value">update value.</param>
    /// <returns>new store call.</returns>
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
    /// build for loop.
    /// </summary>
    /// <param name="loopVar">out index var.</param>
    /// <param name="domain">ranges.</param>
    /// <param name="mode">loop mode.</param>
    /// <param name="var_name">loop var name.</param>
    /// <returns> for builder. </returns>
    public static ISequentialBuilder<For> ForLoop(out Var loopVar, Range domain, LoopMode mode, [CallerArgumentExpression("loopVar")] string var_name = "v")
    {
        var newLoopVar = loopVar = new Var(var_name.StartsWith("var ") ? var_name[4..] : var_name, TensorType.Scalar(DataTypes.Int32));
        return new SequentialBuilder<For>(body => new For(newLoopVar, domain, mode, body));
    }

    /// <summary>
    /// get the Serial For.
    /// </summary>
    /// <param name="loopVar">out index var.</param>
    /// <param name="domain">ranges.</param>
    /// <param name="var_name">loop var name.</param>
    /// <returns> the for loop. </returns>
    public static ISequentialBuilder<For> Serial(out Var loopVar, Range domain, [CallerArgumentExpression("loopVar")] string var_name = "v") => ForLoop(out loopVar, domain, LoopMode.Serial, var_name);

    /// <summary>
    /// make unroll for loop.
    /// </summary>
    /// <param name="loopVar">out index var.</param>
    /// <param name="domain">ranges.</param>
    /// <param name="var_name">loop var name.</param>
    public static ISequentialBuilder<For> Unrolled(out Var loopVar, Range domain, [CallerArgumentExpression("loopVar")] string var_name = "v") => ForLoop(out loopVar, domain, LoopMode.Unrolled, var_name);

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
    [System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.NamingRules", "SA1316:Tuple element names should use correct casing", Justification = "Naming i, j is conventional.")]
    public static ISequentialBuilder<For> Grid(out Var i, out Var j, (Expr i, Expr j) ends)
    {
        var builder_i = T.Serial(out i, (0, ends.i));
        var builder_j = T.Serial(out j, (0, ends.j));
        return new NestBodyExprBuilder<For>(builder_i, builder_j);
    }

    /// <summary>
    /// make the the grid by ranges.
    /// </summary>
    public static ISequentialBuilder<For> Grid(out Var[] loopVars, LoopMode loopMode, params TIR.Range[] ranges)
    {
        string[] names = { "i", "j", "k", "l" };
        var newLoopVars = loopVars = new Var[ranges.Length];
        return new NestBodyExprBuilder<For>(ranges.Select((rg, i) =>
             T.ForLoop(out newLoopVars[i], rg, loopMode, names[i % 4] + (i / 4 == 0 ? string.Empty : (i / 4).ToString())).Body()).ToArray());
    }

    /// <summary>
    /// a named variable represents a tensor index size.
    /// </summary>
    public static Var SizeVar(string name)
    {
        return Var.SizeVar(name);
    }

    /// <summary>
    /// create a block.
    /// </summary>
    public static IBlockBuilder Block(string name)
    {
        return new BlockBuilder(name);
    }

    public static Sequential Sequential(params Expr[] fields)
    {
        return new Sequential(fields);
    }

    public static Sequential Sequential(params object[] fields)
    {
        return TIR.Sequential.Flatten(fields);
    }

    public static ISequentialBuilder<Sequential> Sequential()
    {
        return new SequentialBuilder<Sequential>(body => body);
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
    public static ISequentialBuilder<PrimFunction> PrimFunc(string name, string module_kind, params PhysicalBuffer[] parameters)
    {
        return new SequentialBuilder<PrimFunction>(body => new PrimFunction(name, module_kind, body, parameters));
    }

    /// <summary>
    /// create the handle var.
    /// </summary>
    public static Var Handle(string name, DataType dtype)
    {
        return Var.Handle(name, dtype);
    }

    /// <summary>
    /// rethen the IfThenElseBuilder.
    /// </summary>
    public static IIfThenElseBuilder If(Expr condition)
    {
        return new IfThenElseBuilder(condition);
    }

    /// <summary>
    /// create the memRef by tensortype.
    /// </summary>
    public static LogicalBuffer Buffer(DataType elem_type, Schedule.MemoryLocation location, ReadOnlySpan<Expr> dimensions, out LogicalBuffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        buffer = new LogicalBuffer(name, elem_type, location, dimensions);
        return buffer;
    }

    /// <summary>
    /// ctor for physical buffer.
    /// </summary>
    public static PhysicalBuffer PhysicalBuffer(DataType elem_type, Schedule.MemoryLocation location, ReadOnlySpan<int> dimensions, out PhysicalBuffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        buffer = new PhysicalBuffer(name, elem_type, location, dimensions, 0, (int)TensorUtilities.GetProduct(dimensions.ToArray()) * elem_type.SizeInBytes);
        return buffer;
    }

    /// <summary>
    /// create buffer from const.
    /// </summary>
    public static PhysicalBuffer ConstBuffer(Const expr, out PhysicalBuffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        int size;
        if (expr is TensorConst tc)
        {
            size = tc.Value.BytesBuffer.Length;
        }
        else
        {
            throw new NotSupportedException();
        }

        buffer = new PhysicalBuffer(name, Schedule.MemoryLocation.Rdata, (TensorConst)expr, 0, size);
        return buffer;
    }

#if false
    /// <summary>
    /// maybe can get the const.
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="buffer"></param>
    /// <param name="name"></param>
    /// <returns></returns>
    public static Expr MayBeConst(Const? expr, out Buffer? buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (expr is null)
        {
            buffer = null;
            return Nop();
        }
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }
        buffer = new Buffer(name, Schedule.MemoryLocation.Rdata, (TensorType)expr.ValueType)
        {
            Const = expr,
        };
        return buffer;
    }
#endif

    public static ISequentialBuilder<For> ForSegment(out (Expr B, Expr E) seg, Expr low, Expr chunck, Expr high)
    {
        var count = IR.F.Tensors.Cast((high - low) / IR.F.Tensors.Cast(chunck, DataTypes.Float32), DataTypes.Int32);
        var forloop = T.Serial(out var i, (0, count));
        seg = (i * chunck, IR.F.Math.Min((i + 1) * chunck, high));
        return forloop;
    }

    /// <summary>
    /// Let bind.
    /// </summary>
    /// <param name="v">Variable.</param>
    /// <param name="expression">the expression.</param>
    /// <param name="name">the var name.</param>
    /// <returns>let builder.</returns>
    public static ISequentialBuilder<Let> Let(out Var v, Expr expression, [CallerArgumentExpression("v")] string name = "")
    {
        var newV = v = new Var(name.StartsWith("var ") ? name[4..] : name);
        return new SequentialBuilder<Let>(body => new Let(newV, expression, body));
    }

    /// <summary>
    /// we can use it get some temp var.
    /// </summary>
    public static Call Emit<T>(out T value, Func<T> creator)
    {
        value = creator();
        return Nop();
    }
}
