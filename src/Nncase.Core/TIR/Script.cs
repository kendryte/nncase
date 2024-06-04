// Copyright (c) Canaan Inc. All rights reserved.
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
    public static Call Load(Expr handle, Expr index) => new Call(new Load(), handle, index);

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
    public static Call Store(Expr handle, Expr index, Expr value) => new Call(new Store(), handle, index, value);

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
        var newLoops = ranges.Select((rg, i) => T.ForLoop(out newLoopVars[i], rg, loopMode, names[i % 4] + (i / 4 == 0 ? string.Empty : (i / 4).ToString())).Body()).ToArray();
        return new NestBodyExprBuilder<For>(newLoops);
    }

    public static ISequentialBuilder<For> Grid(out Var[] loopVars, out ISequentialBuilder<For>[] loops, LoopMode loopMode, params TIR.Range[] ranges)
    {
        string[] names = { "i", "j", "k", "l" };
        var newLoopVars = loopVars = new Var[ranges.Length];
        var newLoops = loops = ranges.Select((rg, i) => T.ForLoop(out newLoopVars[i], rg, loopMode, names[i % 4] + (i / 4 == 0 ? string.Empty : (i / 4).ToString())).Body()).ToArray();
        return new NestBodyExprBuilder<For>(loops);
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
    public static ISequentialBuilder<PrimFunction> PrimFunc(string name, string module_kind, params Buffer[] parameters)
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
    /// create the buffer by tensortype.
    /// </summary>
    public static Buffer CreateBuffer(TensorType tensorType, MemoryLocation location, out Buffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        var dimensions = tensorType.Shape.ToValueArray();
        var strides = TensorUtilities.GetStrides(dimensions);
        var size = TensorUtilities.GetProduct(dimensions.ToArray()) * tensorType.DType.SizeInBytes;
        var memspan = new MemSpan(size, location);
        buffer = new Buffer(name, tensorType.DType, memspan, dimensions.Select(i => (Expr)i).ToArray(), strides.Select(i => (Expr)i).ToArray());
        return buffer;
    }

    /// <summary>
    /// create the buffer by expressions.
    /// </summary>
    public static Buffer CreateBuffer(DataType dataType, Expr[] dimensions, MemoryLocation location, out Buffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        var strides = TensorUtilities.GetStrides(dimensions);
        var size = TensorUtilities.GetProduct(dimensions.ToArray()) * dataType.SizeInBytes;
        var memspan = new MemSpan(size, location);
        buffer = new Buffer(name, dataType, memspan, dimensions, strides);
        return buffer;
    }

    public static Buffer CreateBuffer(DataType dataType, Expr[] dimensions, Expr[] strides, MemSpan memSpan, out Buffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        buffer = new Buffer(name, dataType, memSpan, dimensions, strides);
        return buffer;
    }

    public static Buffer AttachBuffer(Expr start, TensorType tensorType, MemoryLocation location, out Buffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        var dimensions = tensorType.Shape.ToValueArray();
        var strides = TensorUtilities.GetStrides(dimensions);
        var size = TensorUtilities.GetProduct(dimensions.ToArray()) * tensorType.DType.SizeInBytes;
        var memspan = new MemSpan(start, size, location);
        buffer = new Buffer(name, tensorType.DType, memspan, dimensions.Select(i => (Expr)i).ToArray(), strides.Select(i => (Expr)i).ToArray());
        return buffer;
    }

    /// <summary>
    /// create buffer by const.
    /// </summary>
    public static Buffer AttachBuffer(TensorConst @const, out Buffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        var dimensions = @const.CheckedShape.ToValueArray();
        var strides = TensorUtilities.GetStrides(dimensions);
        var size = TensorUtilities.GetProduct(dimensions.ToArray()) * @const.CheckedDataType.SizeInBytes;
        var memspan = new MemSpan(IR.F.Buffer.DDrOf(@const), size, MemoryLocation.Rdata);
        buffer = new Buffer(name, @const.CheckedDataType, memspan, dimensions.Select(i => (Expr)i).ToArray(), strides.Select(i => (Expr)i).ToArray());
        return buffer;
    }

    /// <summary>
    /// attach the buffer.
    /// </summary>
    public static Buffer AttachBuffer(Buffer originBuffer, Expr offset, TensorType tensorType, out Buffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        var dimensions = tensorType.Shape.ToValueArray();
        var strides = TensorUtilities.GetStrides(dimensions);
        var size = TensorUtilities.GetProduct(dimensions.ToArray()) * tensorType.DType.SizeInBytes;
        buffer = new Buffer(name, tensorType.DType, originBuffer.MemSpan.SubSpan(offset, size), dimensions.Select(i => (Expr)i).ToArray(), strides.Select(i => (Expr)i).ToArray());
        return buffer;
    }

    /// <summary>
    /// attach the buffer.
    /// </summary>
    public static Buffer AttachBuffer(TensorType tensorType, MemoryLocation location, out Var @var, out Buffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
        {
            name = name[4..];
        }

        @var = new Var(TensorType.Pointer(tensorType.DType));
        var dimensions = tensorType.Shape.ToValueArray();
        var strides = TensorUtilities.GetStrides(dimensions);
        var size = TensorUtilities.GetProduct(dimensions.ToArray()) * tensorType.DType.SizeInBytes;
        buffer = new Buffer(name, tensorType.DType, new MemSpan(@var, size, location), dimensions.Select(i => (Expr)i).ToArray(), strides.Select(i => (Expr)i).ToArray());
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
        buffer = new Buffer(name, MemoryLocation.Rdata, (TensorType)expr.ValueType)
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

    /// <summary>
    /// buffer load.
    /// </summary>
    /// <param name="buffer"> buffer. </param>
    /// <param name="indices"> indices. </param>
    /// <returns> call bufferload. </returns>
    public static Call BufferLoad(TIR.Buffer buffer, params Expr[] indices) => new Call(new IR.Buffers.BufferLoad(), buffer, new IR.Tuple(indices));

    /// <summary>
    /// buffer store.
    /// </summary>
    /// <param name="buffer">buffer.</param>
    /// <param name="indices">indices.</param>
    /// <param name="value">value.</param>
    /// <returns> call bufferstore.</returns>
    public static Call BufferStore(TIR.Buffer buffer, Expr[] indices, Expr value) => new Call(new IR.Buffers.BufferStore(), buffer, new IR.Tuple(indices), value);

    public static Call MatchBuffer(TIR.Buffer buffer) => new Call(new IR.Buffers.MatchBuffer(), buffer);
}
