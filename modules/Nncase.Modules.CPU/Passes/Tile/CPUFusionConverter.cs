// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if true
using System.Reactive;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.CPU;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.Passes.Mutators;
using Nncase.PatternMatch;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Buffer = Nncase.TIR.Buffer;
using MathF = Nncase.IR.F.Math;
using Range = Nncase.TIR.Range;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Passes.Tile;

/// <summary>
/// name 分配器.
/// </summary>
internal sealed class NameAllocator
{
    public Dictionary<string, int> NamePool { get; } = new();

    public string Get(string name)
    {
        if (!NamePool.TryGetValue(name, out var count))
        {
            count = 0;
        }

        NamePool[name] = count + 1;
        return count == 0 ? name : $"{name}_{count}";
    }
}

internal class CPUFusionConverter
{
    public NameAllocator NameAllocator { get; } = new();

    /// <summary>
    /// Gets tile size 的变量.
    /// </summary>
    public List<Var> TileSizeVars { get; } = new();

    /// <summary>
    /// Gets loop 变量.
    /// </summary>
    // public List<Var> LoopVars { get; } = new();

    /// <summary>
    /// Gets loop domains.
    /// </summary>
    public List<Range> LoopDomains { get; } = new();

    /// <summary>
    /// Gets nested loops.
    /// </summary>
    // public List<ISequentialBuilder<For>> NestedLoops { get; } = new();

    public TileOptions TileOptions { get; protected set; } = null!;

    /// <summary>
    /// Gets or sets 总的loop count.
    /// </summary>/
    public virtual Expr LoopCount { get; protected set; }

    /// <summary>
    /// Gets or sets ping pong 外层的tiling.
    /// </summary>
    public virtual Expr LoopCountOuter { get; protected set; }

    /// <summary>
    /// Gets or sets ping pong 内侧的tiling.
    /// </summary>
    public virtual Expr LoopCountInner { get; protected set; }

    /// <summary>
    /// Gets or sets 当前的fusion.
    /// </summary>
    public virtual Fusion CurrentFusion { get; protected set; }

    public virtual PrimFunction BuildPrimFunc(Fusion fusion)
    {
        // TODO: buffer顺序可能需要调整以保持原图的顺序
        var primFuncBuilder = T.PrimFunc(CurrentFusion.Name, CPUTarget.Kind, _ifBufferMap.Values.Union(_ofBufferMap.Values).Select(b => (PhysicalBuffer)b).ToArray());
        return primFuncBuilder.Build();
    }

    public virtual Expr Visit(Expr root)
    {
        return root switch
        {
            Call call => (call.Target switch
            {
                CPUUnary op => LowerCPUUnary(call, op),
                _ => throw new NotSupportedException(),
            }).Build(),
            _ => T.Nop(),
        };
    }

    protected virtual ISequentialBuilder<Sequential> LowerCPUUnary(Call call, CPUUnary op)
    {
        var prefix = NameAllocator.Get(nameof(CPUUnary));
        var inputCall = call[CPUUnary.Input];
        T.PhysicalBuffer(inputCall.CheckedDataType, MemoryLocation.Input, inputCall.CheckedShape, out var ddrIf);
        T.PhysicalBuffer(call.CheckedDataType, MemoryLocation.Output, call.CheckedShape.ToValueArray(), out var ddrOf);
        _ifBufferMap.Add(call, ddrIf);
        _ofBufferMap.Add(call, ddrOf);

        List<Var> LoopVars = new();
        List<ISequentialBuilder<For>> NestedLoops = new();
        List<Range> LoopDomains = call.CheckedShape.Select(s => new Range(0, 1, s.FixedValue)).ToList();

        var seq = T.Sequential().Body(Visit(inputCall));

        for (int i = 0; i < call.CheckedShape.Rank; i++)
        {
            NestedLoops.Add(T.ForLoop(out var loopVar, LoopDomains[i], LoopMode.Unrolled, $"loop_var_{i}"));
            LoopVars.Add(loopVar);
        }

        NestedLoops[^1].Body(
            op.UnaryOp switch
            {
                // TODO: body的实现
                UnaryOp.Abs => T.Nop(),
                _ => throw new NotSupportedException(),
            });

        seq.Body(NestedLoops[0].Body());
        return seq;
    }

    private readonly Dictionary<Expr, Buffer> _ifBufferMap = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<Expr, Buffer> _ofBufferMap = new(ReferenceEqualityComparer.Instance);
}
#endif
