// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;


/// <summary>
/// PrimFunction expression.
/// </summary>
public sealed record PrimFunction(string Name, Sequential Body, IRArray<Buffer> Parameters) : Callable(Name), ISequentialExpr
{
    private static int _globalFuncIndex = 0;

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// </summary>
    /// <param name="parameters">Parameters.</param>
    /// <param name="body">Body.</param>
    public PrimFunction(Sequential body, IRArray<Buffer> parameters)
        : this($"primfunc_{_globalFuncIndex++}", body, parameters)
    {
    }

    /// <summary>
    /// build function.
    /// </summary>
    /// <param name="body"></param>
    /// <param name="parameters"></param>
    public PrimFunction(Sequential body, params Buffer[] parameters)
        : this($"primfunc_{_globalFuncIndex++}", body, new(parameters))
    {
    }
}


/// <summary>
/// The container of Exprs.
/// Represent a sequence of Expr.
/// </summary>
public sealed record Sequential : Expr, IList<Expr>
{
    /// <summary>
    /// the sub Fields
    /// </summary>
    public IRArrayList<Expr> Fields;

    /// <summary>
    /// ctor for default.
    /// </summary>
    /// <param name="fields">sub fields.</param>
    public Sequential(IRArrayList<Expr> fields)
    {
        Fields = new(Flatten(fields));
    }

    /// <summary>
    /// Sequential ctor.
    /// </summary>
    public Sequential() : this(new IRArrayList<Expr>()) { }

    /// <summary>
    /// get the fields.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public Expr this[int index] { get => ((IList<Expr>)Fields)[index]; set => ((IList<Expr>)Fields)[index] = value; }

    /// <summary>
    /// flatten the exprs
    /// </summary>
    /// <param name="exprs"></param>
    /// <returns></returns>
    public static List<Expr> Flatten(IEnumerable<Expr> exprs)
    {
        var ret = new List<Expr>();
        foreach (var item in exprs)
        {
            if (item is Sequential sub)
                ret.AddRange(Flatten(sub));
            else
                ret.Add(item);
        }
        return ret;
    }

    /// <inheritdoc/>
    public int Count => ((ICollection<Expr>)Fields).Count;

    /// <inheritdoc/>
    public bool IsReadOnly => ((ICollection<Expr>)Fields).IsReadOnly;

    /// <inheritdoc/>
    public void Add(Expr item) => Fields.Add(item);

    /// <inheritdoc/>
    public void Clear()
    {
        ((ICollection<Expr>)Fields).Clear();
    }

    /// <inheritdoc/>
    public bool Contains(Expr item)
    {
        return ((ICollection<Expr>)Fields).Contains(item);
    }

    /// <inheritdoc/>
    public void CopyTo(Expr[] array, int arrayIndex)
    {
        ((ICollection<Expr>)Fields).CopyTo(array, arrayIndex);
    }

    /// <inheritdoc/>
    public IEnumerator<Expr> GetEnumerator()
    {
        return ((IEnumerable<Expr>)Fields).GetEnumerator();
    }

    /// <inheritdoc/>
    public int IndexOf(Expr item)
    {
        return ((IList<Expr>)Fields).IndexOf(item);
    }

    /// <inheritdoc/>
    public void Insert(int index, Expr item)
    {
        ((IList<Expr>)Fields).Insert(index, item);
    }

    /// <inheritdoc/>
    public bool Remove(Expr item)
    {
        return ((ICollection<Expr>)Fields).Remove(item);
    }

    /// <inheritdoc/>
    public void RemoveAt(int index)
    {
        ((IList<Expr>)Fields).RemoveAt(index);
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return ((IEnumerable)Fields).GetEnumerator();
    }
}

/// <summary>
/// select the value and return it, the true and false must have same type!.
/// </summary>
/// <param name="Condition"></param>
/// <param name="TrueValue"></param>
/// <param name="FalseValue"></param>
public sealed record Select(Expr Condition, Expr TrueValue, Expr FalseValue) : Expr { }

/// <summary>
/// Load value from the result produced by the producer.
/// </summary>
/// <remarks>
/// This node only appears in high-level DSLs that are built on top of the TIR.
/// It should not appear in a valid TIR PrimFunc. A high-level DSL needs to lower
/// this node before TIR transformations.
/// </remarks>
/// <seealso cref="ProducerLoad"/>
/// <param name="Producer">The buffer producer.</param>
/// <param name="Indices">The location arguments.</param>
public sealed record ProducerLoad(DataProducer Producer, IRArray<Expr> Indices) : Expr
{
}

/// <summary>
/// Create a vector where all the elements are value.
/// </summary>
/// <param name="Value"> The base value. </param>
/// <param name="lanes"> The number of lanes. </param>
public sealed record Broadcast(Expr Value, int lanes) : Expr
{
}

/// <summary>
/// Let binding. Bind var to value then evaluate body. return unit.
/// </summary>
/// <param name="Var"> The expr . </param>
/// <param name="Expression"> The value to be binded. </param>
/// <param name="Body"> The Let body. </param>
public sealed record Let(Buffer Var, Expr Expression, Sequential Body) : Expr, ISequentialExpr
{
}

/// <summary>
/// A While loop.
/// <code>
///   while (condition) { body }
/// </code>
/// </summary>
/// <param name="Condition">The termination condition.</param>
/// <param name="Bodys">The body of the while loop.</param>
public sealed record While(Expr Condition, Sequential Bodys) : Expr
{ }

/// <summary>
/// The interface mean the expr has body
/// </summary>
public interface ISequentialExpr
{
    /// <summary>
    /// The body of the for loop.
    /// </summary>
    public Sequential Body { get; }
}

/// <summary>
/// A for loop, with poissible type annotations.
/// <example>
/// <code>
///   for (loop_var = min; loop_var { min + extent; ++loop_var) {
///     body
///    }
/// </code>
/// </example>
/// </summary>
/// <param name="LoopVar">The loop variable.</param>
/// <param name="Dom">The dom of for range.</param>
/// <param name="Mode">The kind of the for loop.</param>
/// <param name="Body">the body sequence.</param>
public sealed record For(Var LoopVar, Range Dom, LoopMode Mode, Sequential Body) : Expr, ISequentialExpr
{
    /// <summary>
    /// ctor.
    /// </summary>
    /// <param name="LoopVar"></param>
    /// <param name="Dom"></param>
    /// <param name="Mode"></param>
    public For(Var LoopVar, Range Dom, LoopMode Mode) : this(LoopVar, Dom, Mode, new()) { }

    /// <summary>
    /// implcit cast to Var, so we can get the Loop it selp and it's loop var.
    /// </summary>
    /// <param name="loop"></param>
    public static implicit operator Var(For loop) => loop.LoopVar;
}

/// <summary>
/// Shuffle instruction.
///  vec = concat(vectors)
///  result = (vec[indices[0]], vec[indices[1]] ...)
/// </summary>
/// <param name="Vectors">the input vectors. </param>
/// <param name="Indices">The indices of each element. </param>
public sealed record Shuffle(IRArray<Expr> Vectors, IRArray<Expr> Indices) : Expr
{
}

/// <summary>
/// A commutative reducer node to represent a commutative
///  binary operator with identity element.
/// </summary>
/// <param name="Lhs">The left argument of reducer. </param>
/// <param name="Rhs">The right argument of reducer. </param>
/// <param name="Result">The result of reducer. </param>
/// <param name="IdentityElement">The identity element of reducer, which leaves other
///  elements unchanged when combined with it, with respect to
///  the binary operation of this reducer uses. </param>
public sealed record CommReducer(IRArray<Var> Lhs, IRArray<Var> Rhs, IRArray<Expr> Result, IRArray<Expr> IdentityElement)
{
    /// <summary>
    /// Function call operator to combine a and b.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns></returns>
    public IRArray<Expr> Combine(IRArray<Expr> a, IRArray<Expr> b) { return a; }


    /// <inheritdoc/>
    public override string ToString()
    {
        return $"comm_reducer(result={Result}, lhs= {Lhs}, rhs= {Rhs}, identity_element= {IdentityElement})";
    }
}

/// <summary>
/// Reduction operator.
/// </summary>
public sealed record Reduction : Expr
{
    /// <summary>
    /// The source operand.
    /// </summary>
    public CommReducer? Combiner;

    /// <summary>
    /// The reduction axis.
    /// </summary>
    public IRArray<Expr> Source;

    /// <summary>
    /// Predicate on the reduction Only add the body to reduction if condition is true.
    /// </summary>
    public Expr Condition;

    /// <summary>
    /// The init operand.
    /// </summary>
    public IRArray<Expr>? Init;

    /// <summary>
    /// the index of this reduce node.
    /// </summary>
    public IRArray<IterVar> Axis;

    /// <summary>
    /// The commutative combiner.
    /// </summary>
    public int ValueIndex;

    /// <summary>
    ///
    /// </summary>
    /// <param name="combiner"></param>
    /// <param name="source"></param>
    /// <param name="axis"></param>
    /// <param name="condition"></param>
    /// <param name="value_index"></param>
    /// <param name="init"></param>
    /// <exception cref="InvalidOperationException"></exception>
    public Reduction(CommReducer? combiner, IRArray<Expr> source, IRArray<IterVar> axis,
           Expr? condition = null, int value_index = 0, IRArray<Expr>? init = null)
    {
        if (!axis.All(x => x.Mode == IterationMode.CommReduce))
        {
            throw new InvalidOperationException("Can only take axis created by reduce_axis");
        }

        if (condition is null)
        {
            condition = (Const)1;
        }

        if (init is not null)
        {
            if (source.Count != init?.Count)
            {
                throw new InvalidOperationException("");
            }

            if (!init.Value.All(x => (
              x is ProducerLoad) ||
               ((x is Const con) &&
                (TypePatternUtility.IsFloatScalar().MatchLeaf(con.CheckedType!) || TypePatternUtility.IsIntegralScalar().MatchLeaf(con.CheckedType!)))))
            {
                throw new InvalidOperationException("init can only be a IntImm, FloatImm or ProducerLoad");
            }
        }

        Combiner = combiner;
        Source = source;
        Axis = axis;
        Condition = condition;
        ValueIndex = value_index;
        Init = init;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"reduction(combiner= {Combiner}, source= {Source}, init= {Init}, axis= {Axis}, where= {Condition}, value_index= {ValueIndex})";
    }
}

/// <summary>
/// Representing the region of multi-dimensional buffer access.
/// </summary>
/// <param name="Buffer">The buffer of the buffer region.</param>
/// <param name="Region">The region array of the buffer region.</param>
public sealed record BufferRegion(Buffer Buffer, IRArray<Range> Region)
{
    /// <summary>
    /// Create a BufferRegion which is full region of the given buffer.
    /// </summary>
    /// <param name="Buf">The buffer to generate full BufferRegion.</param>
    /// <returns>The BufferRegion which covers all region of the given buffer.</returns>
    public static BufferRegion Full(Buffer Buf) => new BufferRegion(Buf, new(Buf.Shape.ToArray().Select(extent => new Range(0, extent, 1))));

    /// <summary>
    /// Create a BufferRegion which is a single point of the given buffer.
    /// </summary>
    /// <param name="Buf">The buffer to generate single point BufferRegion.</param>
    /// <param name="Indices">The access point indices of the buffer.</param>
    /// <returns>The BufferRegion which is the single point of the given buffer.</returns>
    public static BufferRegion FromPoint(Buffer Buf, IRArray<Expr> Indices) => new BufferRegion(Buf, new(Indices.Select(index => new Range(index, index + 1, 1))));
}

/// <summary>
/// Match introduces a constraint that the source buffer region can be remapped to the data
/// layout specified by the buffer field. The constraint can be checked in later part of lowering (or
/// optionally during runtime).
///
/// MatchBufferRegion provides a mechanism to represent data layout and compactness constraints in
/// low-level hardware primitives in the IR and defer the check after the sequence of
/// transformations.
/// </summary>
/// <param name="Buffer">The target buffer.</param>
/// <param name="Source">The source buffer region.</param>
public sealed record MatchBufferRegion(Buffer Buffer, BufferRegion Source)
{ }

/// <summary>
/// A block is a basic schedule unit in TIR.
/// <remarks>
/// Block's body is parameterized by iter vars.
/// </remarks>
/// <code>
///   with T.block(name):
///   v0 = T.axis.S(domain, value0)
///   v1 = T.axis.R(domain, value1)
///   ...
///   T.reads([buffer0[start:end, ...], ...])
///   T.writes([buffer1[start:end, ...], ...])
///   T.where(predicate)
///   buffer2 = T.alloc_buffer(shape, dtype)
///   buffer3 = T.match_buffer(source_buffer[start:end, ...])
///   T.attr({attr_key: attr_value, ...})
///   with T.init():
///      init body
///    body
/// </code>
/// </summary>
/// <param name="Name"> The name_hint of the block.</param>
/// <param name="Body"> block body </param>
/// <param name="InitBody">the Block init statement.</param>
/// <param name="IterVars">The List Exprs contain the IterVars</param>
/// <param name="Reads">The read buffer regions of the block.</param>
/// <param name="Writes">The write buffer regions of the block.</param>
/// <param name="AllocBuffers">The buffer allocated in the block.</param>
/// <param name="Predicate">The predicate of the block realization, the block will only be executed when the predicate is true.</param>
public sealed record Block(string Name, Sequential Body, Sequential InitBody,
                            IRArrayList<IterVar> IterVars,
                            IRArrayList<BufferRegion> Reads,
                            IRArrayList<BufferRegion> Writes,
                            IRArrayList<Buffer> AllocBuffers, Expr Predicate) : Expr, ISequentialExpr
{
    /// <summary>
    /// <see cref="Block"/>.
    /// </summary>
    /// <param name="Name">block name.</param>
    public Block(string Name) : this(Name, new(), new(), new(), new(), new(), new(), true) { }

    /// <summary>
    /// bind the itervar with for loop.
    /// </summary>
    /// <param name="vi"></param>
    /// <param name="fi"></param>
    /// <param name="iter_type"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    public Block Remap(out IterVar vi, For fi, char iter_type)
    {
        var toMode = (char x) => x switch
        {
            'S' => IterationMode.DataParallel,
            'R' => IterationMode.CommReduce,
            _ => throw new NotSupportedException("Only Support \"S\" (for Spatial) or \"R\" ( Reduce)"),
        };
        return Bind(out vi, fi.Dom, toMode(iter_type), fi.LoopVar);
    }

    /// <inheritdoc/>
    public Block Remap(out IterVar vi, out IterVar vj, (For i, For j) loops, string iter_types)
    {
        return Remap(out vi, loops.i, iter_types[0]).
        Remap(out vj, loops.j, iter_types[1]);
    }

    /// <summary>
    /// create the iterVar and bind the value.
    /// </summary>
    /// <param name="vi"></param>
    /// <param name="dom"></param>
    /// <param name="mode"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public Block Bind(out IterVar vi, Range dom, IterationMode mode, Expr value)
    {
        vi = new IterVar(TensorType.Scalar(DataTypes.Int32), dom, mode, value);
        IterVars.Add(vi);
        return this;
    }

    /// <summary>
    /// set the init feilds.
    /// </summary>
    /// <param name="exprs"></param>
    /// <returns></returns>
    public T.SequentialBuilder<Block> Init(params Expr[] exprs)
    {
        foreach (var item in exprs)
        {
            InitBody.Add(item);
        }
        return new(this);
    }
}

/// <summary>
/// Buffer store node.
/// </summary>
/// <param name="Buffer">The buffer.</param>
/// <param name="Indices">The value we to be stored.</param>
/// <param name="Value">The indices location to be stored.</param>
public sealed record BufferStore(Buffer Buffer, IRArray<Expr> Indices, Expr Value) : Expr
{
}

/// <summary>
/// Buffer load node.
/// </summary>
/// <param name="Buffer">The buffer to be loaded.</param>
/// <param name="Indices">The buffer indices.</param>
public sealed record BufferLoad(Buffer Buffer, IRArray<Expr> Indices) : Expr
{
}

/// <summary>
/// if(xxx) then { zzz } else { yyy }.
/// </summary>
/// <param name="Condition"></param>
/// <param name="Then"> Sequential. </param>
/// <param name="Else"> Sequential. </param>
public sealed record IfThenElse(Expr Condition, Expr Then, Expr Else) : Expr
{

    /// <summary>
    /// ctor.
    /// </summary>
    /// <param name="Condition"></param>
    /// <param name="Then"></param>
    public IfThenElse(Expr Condition, Expr Then) : this(Condition, Then, new Sequential()) { }
}
