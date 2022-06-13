// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.


using System.Runtime.CompilerServices;
using Nncase.IR;

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
    /// get the nop op
    /// </summary>
    /// <returns></returns>
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
    /// build the sequential
    /// </summary>
    /// <typeparam name="T"></typeparam>
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
    public class SequentialBuilder<T> : ISequentialBuilder<T>
      where T : ISequentialExpr
    {
        /// <summary>
        /// expr 
        /// </summary>
        public T ParentExpr;

        /// <summary>
        /// ctor
        /// </summary>
        /// <param name="expr"></param>
        public SequentialBuilder(T expr)
        {
            ParentExpr = expr;
        }

        /// <summary>
        /// Add the expr items to body
        /// </summary>
        /// <param name="exprs"></param>
        /// <returns></returns>
        public T Body(params Expr[] exprs)
        {
            TIR.Sequential.Flatten(exprs).ForEach(e => ParentExpr.Body.Add(e));
            return ParentExpr;
        }
    }

    /// <summary>
    /// build for loop.
    /// </summary>
    /// <param name="loopVar">out index var.</param>
    /// <param name="Dom">ranges.</param>
    /// <param name="mode">loop mode.</param>
    /// <param name="loop">loop instance.</param>
    /// <param name="var_name">loop var name.</param>
    /// <returns> for builder. </returns>
    public static SequentialBuilder<For> ForLoop(out Var loopVar, Range Dom, LoopMode mode, out For loop, [CallerArgumentExpression("loopVar")] string var_name = "v")
    {
        loopVar = new Var(var_name.StartsWith("var ") ? var_name[4..] : var_name, TensorType.Scalar(DataTypes.Int32));
        loop = new For(loopVar, Dom, mode);
        return new SequentialBuilder<For>(loop);
    }

    /// <summary>
    /// get the Serial For
    /// </summary>
    /// <param name="loopVar">out index var.</param>
    /// <param name="Dom">ranges.</param>
    /// <param name="loop">loop instance.</param>
    /// <param name="var_name">loop var name.</param>
    /// <returns> the for loop </returns>
    public static SequentialBuilder<For> Serial(out Var loopVar, Range Dom, out For loop, [CallerArgumentExpression("loopVar")] string var_name = "v") => ForLoop(out loopVar, Dom, LoopMode.Serial, out loop, var_name);

    /// <summary>
    /// serial
    /// </summary>
    /// <param name="loopVar">out index var.</param>
    /// <param name="Dom">ranges.</param>
    /// <param name="var_name">loop var name.</param>
    /// <returns></returns>
    public static SequentialBuilder<For> Serial(out Var loopVar, Range Dom, [CallerArgumentExpression("loopVar")] string var_name = "v") => Serial(out loopVar, Dom, out _, var_name);

    /// <summary>
    /// make unroll for loop
    /// </summary>
    /// <param name="loopVar">out index var.</param>
    /// <param name="Dom">ranges.</param>
    /// <param name="loop">ranges.</param>
    /// <param name="var_name">loop var name.</param>
    /// <returns></returns>
    public static SequentialBuilder<For> Unrolled(out Var loopVar, Range Dom, out For loop, [CallerArgumentExpression("loopVar")] string var_name = "v") => ForLoop(out loopVar, Dom, LoopMode.Unrolled, out loop, var_name);

    /// <summary>
    /// GridWrapper for collect the for item.
    /// </summary>
    public class NestBodyExprBuilder<T> : ISequentialBuilder<T>
     where T : Expr, ISequentialExpr
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
                exprs[i].Body.Add(exprs[i + 1]);
            }
            Exprs = exprs;
        }

        /// <summary>
        /// Wrapper Body method
        /// </summary>
        /// <param name="exprs"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public T Body(params Expr[] exprs)
        {
            Sequential.Flatten(exprs).ForEach(item => Exprs.Last().Body.Add(item));
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
        var builder_i = T.Serial(out i, (0, ends.i), out var for_i);
        var builder_j = T.Serial(out j, (0, ends.j), out var for_j);
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
        T.Serial(out i, (0, ends.i), out loops.i);
        T.Serial(out j, (0, ends.j), out loops.j);
        return new NestBodyExprBuilder<For>(loops.i, loops.j);
    }

    /// <summary>
    /// make the the grid by ranges.
    /// </summary>
    /// <param name="loopMode"></param>
    /// <param name="ranges"></param>
    /// <returns></returns>
    public static NestBodyExprBuilder<For> Grid(LoopMode loopMode, params Range[] ranges)
    {
        string[] names = { "i", "j", "k", "l" };
        return new NestBodyExprBuilder<For>(ranges.Select((rg, i) =>
             T.ForLoop(out var _, rg, loopMode, out var _, names[i % 4] + (i / 4 == 0 ? "" : (i / 4).ToString())).Body()
        ).ToArray());
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
    /// <param name="module_kind"></param>
    /// <param name="parameters"></param>
    /// <returns></returns>
    public static SequentialBuilder<PrimFunction> PrimFunc(string name, string module_kind, params Buffer[] parameters)
    {
        return new(new PrimFunction(name, module_kind, new(), new IRArray<Buffer>(parameters)));
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

    /// <summary>
    /// builfer the if then else block
    /// </summary>
    public class IfThenElseBuilder
    {
        readonly Sequential ThenBranch;
        readonly Sequential ElseBranch;
        readonly Expr Condition;

        /// <summary>
        /// ctor.
        /// </summary>
        /// <param name="condition"></param>
        public IfThenElseBuilder(Expr condition)
        {
            ThenBranch = new();
            ElseBranch = new();
            Condition = condition;
        }

        /// <summary>
        /// then block
        /// </summary>
        /// <param name="exprs"> statements. </param>
        /// <returns> IfThenElseBuilder. </returns>
        public IfThenElseBuilder Then(params Expr[] exprs)
        {
            foreach (var item in exprs)
            {
                ThenBranch.Add(item);
            }

            return this;
        }

        /// <summary>
        /// else block
        /// </summary>
        /// <param name="exprs"> statements. </param>
        /// <returns> IfThenElseBuilder. </returns>
        public IfThenElseBuilder Else(params Expr[] exprs)
        {
            foreach (var item in exprs)
            {
                ElseBranch.Add(item);
            }

            return this;
        }

        /// <summary>
        /// get the expr.
        /// </summary>
        /// <returns>IfThenElse.</returns>
        public IfThenElse ToExpr()
        {
            return new IfThenElse(Condition, ThenBranch, ElseBranch);
        }

        /// <summary>
        /// cast to expr.
        /// </summary>
        /// <param name="builder"></param>
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

    /// <summary>
    /// create the memRef by tensortype.
    /// </summary>
    /// <param name="type"></param>
    /// <param name="buffer"></param>
    /// <param name="name"></param>
    /// <param name="location"></param>
    /// <returns></returns>
    public static Buffer Buffer(TensorType type, Schedule.MemoryLocation location, out Buffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
            name = name[4..];
        buffer = new Buffer(name, location, type);
        return buffer;
    }

    /// <summary>
    /// create buffer from const
    /// </summary>
    /// <param name="expr"></param>
    /// <param name="buffer"></param>
    /// <param name="name"></param>
    /// <returns></returns>
    public static Buffer ConstBuffer(Const expr, out Buffer buffer, [CallerArgumentExpression("buffer")] string name = "")
    {
        if (name.StartsWith("var "))
            name = name[4..];
        buffer = new Buffer(name, Schedule.MemoryLocation.Rdata, (TensorType)expr.ValueType)
        {
            Const = expr
        };
        return buffer;
    }

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
            name = name[4..];
        buffer = new Buffer(name, Schedule.MemoryLocation.Rdata, (TensorType)expr.ValueType)
        {
            Const = expr
        };
        return buffer;
    }

    public static SequentialBuilder<For> ForSegment(out (Expr b, Expr e) seg, Expr low, Expr chunck, Expr high)
    {
        var count = IR.F.Tensors.Cast((high - low) / IR.F.Tensors.Cast(chunck, DataTypes.Float32), DataTypes.Int32);
        var forloop = T.Serial(out var i, (0, count));
        seg = ((i * chunck), IR.F.Math.Min(((i + 1) * chunck), high));
        return forloop;
    }

    /// <summary>
    /// Let bind.
    /// </summary>
    /// <param name="v"></param>
    /// <param name="expression">the expression.</param>
    /// <param name="name">the var name.</param>
    /// <returns>let builder.</returns>
    public static SequentialBuilder<Let> Let(out Var v, Expr expression, [CallerArgumentExpression("v")] string name = "")
    {
        v = new Var(name.StartsWith("var ") ? name[4..] : name);
        var let = new Let(v, expression, new());
        return new(let);
    }

    /// <summary>
    /// we can use it get some temp var.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="value"></param>
    /// <param name="creator"></param>
    /// <returns></returns>
    public static Call Emit<T>(out T value, Func<T> creator)
    {
        value = creator();
        return Nop();
    }

}
