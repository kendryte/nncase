// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.Builders;

/// <summary>
/// builfer the block.
/// </summary>
public interface IBlockBuilder : IExprBuilder<Block>
{
    /// <summary>
    /// else block.
    /// </summary>
    /// <param name="exprOrBuilders"> statements. </param>
    /// <returns> BlockBuilder. </returns>
    IBlockBuilder Body(params object[] exprOrBuilders);

    /// <summary>
    /// then block.
    /// </summary>
    /// <param name="exprOrBuilders"> statements. </param>
    /// <returns> BlockBuilder. </returns>
    IBlockBuilder Init(params object[] exprOrBuilders);

    /// <summary>
    /// create the iterVar and bind the value.
    /// </summary>
    IBlockBuilder Bind(out IterVar vi, Range domain, IterationMode mode, Var value);

    /// <summary>
    /// bind the itervar with for loop.
    /// </summary>
    IBlockBuilder Remap(out IterVar vi, For fi, char iterType);

    /// <summary>
    /// alloctions.
    /// </summary>
    IBlockBuilder Alloc(params object[] buffers);

    /// <summary>
    /// reads.
    /// </summary>
    IBlockBuilder Reads(params object[] buffer_regions);

    /// <summary>
    /// writes.
    /// </summary>
    IBlockBuilder Writes(params object[] buffer_regions);

    IBlockBuilder Predicate(Expr predicate);
}

internal class BlockBuilder : IBlockBuilder
{
    private readonly string _name;
    private readonly List<object> _init = new();
    private readonly List<object> _body = new();
    private readonly List<IterVar> _iterVars = new();
    private readonly List<TIR.Buffer> _allocations = new();
    private readonly List<TIR.BufferRegion> _reads = new();
    private readonly List<TIR.BufferRegion> _writes = new();
    private Expr? _predicate;

    public BlockBuilder(string name)
    {
        _name = name;
    }

    public IBlockBuilder Body(params object[] exprOrBuilders)
    {
        _body.AddRange(exprOrBuilders);
        return this;
    }

    public IBlockBuilder Init(params object[] exprOrBuilders)
    {
        _init.AddRange(exprOrBuilders);
        return this;
    }

    public IBlockBuilder Bind(out IterVar vi, Range dom, IterationMode mode, Var value)
    {
        vi = new IterVar(dom, mode, value);
        _iterVars.Add(vi);
        return this;
    }

    public IBlockBuilder Remap(out IterVar vi, For fi, char iterType)
    {
        var toMode = (char x) => x switch
        {
            'S' => IterationMode.DataParallel,
            'R' => IterationMode.CommReduce,
            _ => throw new NotSupportedException("Only Support \"S\" (for Spatial) or \"R\" ( Reduce)"),
        };
        return Bind(out vi, fi.Domain, toMode(iterType), fi.LoopVar);
    }

    public Block Build()
    {
        return new(_name, Sequential.Flatten(CollectionsMarshal.AsSpan(_body)), Sequential.Flatten(CollectionsMarshal.AsSpan(_init)), CollectionsMarshal.AsSpan(_iterVars), CollectionsMarshal.AsSpan(_reads), CollectionsMarshal.AsSpan(_writes), CollectionsMarshal.AsSpan(_allocations), _predicate ?? true);
    }

    public IBlockBuilder Alloc(params object[] buffers)
    {
        HashSet<TIR.Buffer> set = new(ReferenceEqualityComparer.Instance);
        Add(set, buffers);
        _allocations.AddRange(set.ToList());
        return this;
    }

    public IBlockBuilder Reads(params object[] buffer_regions)
    {
        HashSet<TIR.BufferRegion> set = new(ReferenceEqualityComparer.Instance);
        Add(set, buffer_regions);
        _reads.AddRange(set.ToList());
        return this;
    }

    public IBlockBuilder Writes(params object[] buffer_regions)
    {
        HashSet<TIR.BufferRegion> set = new(ReferenceEqualityComparer.Instance);
        Add(set, buffer_regions);
        _reads.AddRange(set.ToList());
        return this;
    }

    public IBlockBuilder Predicate(Expr predicate)
    {
        _predicate ??= predicate;
        return this;
    }

    private static void Add<T>(HashSet<T> set, IEnumerable<object> inputs)
    {
        if (inputs is null)
        {
            return;
        }

        foreach (var obj in inputs)
        {
            switch (obj)
            {
                case T item:
                    set.Add(item);
                    break;
                case IEnumerable<object> items:
                    Add(set, items);
                    break;
                default:
                    break;
            }
        }
    }
}
