// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.Builders;

namespace Nncase.Schedule.TileTree;

public sealed class TreeSolverArgumentsCollector : ITreeNodeVisitor<Unit, Unit>
{
    public TreeSolverArgumentsCollector()
    {
        Inputs = new();
        Outputs = new();
    }

    public HashSet<BufferIdenitity> Inputs { get; }

    public HashSet<BufferIdenitity> Outputs { get; }

    public Unit Visit(ScopeNode value, Unit arg1)
    {
        foreach (var child in value.Children)
        {
            child.Accept(this, arg1);
        }

        return default;
    }

    public Unit Visit(TileNode value, Unit arg1)
    {
        return value.Child.Accept(this, arg1);
    }

    public Unit Visit(OpNode value, Unit arg1)
    {
        for (int i = 0; i < value.BufferShapes.Length - 1; i++)
        {
            Inputs.Add(new(value, i));
        }

        Outputs.Add(new(value, value.BufferShapes.Length - 1));

        foreach (var dep in value.Dependences)
        {
            Inputs.Remove(new(value, dep.Index));
            Outputs.Remove(new(dep.Node, dep.Node.BufferShapes.Length - 1));
        }

        return default;
    }
}

public sealed class TreeSolverResultConstructor : TreeSolverBase, ITreeNodeVisitor<TreeSolverResultConstructor.Context, Unit>
{
    private readonly Assignment _sol;
    private readonly HashSet<BufferIdenitity> _inputs;
    private readonly HashSet<BufferIdenitity> _outputs;
    private readonly Dictionary<ITileAbleNode, Dictionary<BufferIdenitity, SubViewInfo>> _subViewMemo;

    public TreeSolverResultConstructor(Assignment solution, HashSet<BufferIdenitity> inputs, HashSet<BufferIdenitity> outputs, Solver solver, IntExpr one, IntExpr zero, IntExpr elem, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo> levelBufferInfos, Dictionary<ITileAbleNode, DomainInfo> domainInfos, CompileOptions compileOptions)
        : base(solver, one, zero, elem, primitiveBufferInfo, levelBufferInfos, domainInfos, compileOptions.TargetOptions)
    {
        _sol = solution;
        _inputs = inputs;
        _outputs = outputs;
        CompileOptions = compileOptions;
        BufferMemo = new();
        _subViewMemo = new();
    }

    public Dictionary<BufferIdenitity, TIR.Buffer> BufferMemo { get; }

    public CompileOptions CompileOptions { get; }

    public Unit Visit(ScopeNode value, Context context)
    {
        var (parentbuilder, partentOffsets) = context;
        foreach (var child in value.Children)
        {
            var childBuilder = T.Sequential();
            child.Accept(this, new(childBuilder, partentOffsets));
            parentbuilder.Body(childBuilder);
        }

        return default;
    }

    public Unit Visit(TileNode value, Context context)
    {
        var (parentbuilder, partentOffsets) = context;

        var loopBuilders = new ISequentialBuilder<TIR.For>[value.DimNames.Length];
        var loopVars = new Var[value.DimNames.Length];

        var nodeMemo = TileNodeMemo[value];

        // from inner to outer
        for (int i = value.DimNames.Length - 1; i >= 0; i--)
        {
            long stop = _sol.Value(nodeMemo.BackWardExtents[0][i].Var());
            long tileSize = _sol.Value(TileableNodeMemo[value].TileVars[i]);
            loopBuilders[i] = T.Serial(out var loopVar, (0L, stop, stop / tileSize), TileableNodeMemo[value].TileVars[i].Name());
            loopVars[i] = loopVar;
        }

        var initOffsets = Enumerable.Repeat<Expr>(0L, loopVars.Length).ToArray();
        foreach (var (k, v) in TileableNodeMemo[value].DimsMap)
        {
            initOffsets[k] += partentOffsets[v];
        }

        var forwardOffsets = new Expr[loopVars.Length][];
        for (int i = 0; i < loopVars.Length; i++)
        {
            var offsets = forwardOffsets[i] = initOffsets.ToArray();

            for (int j = 0; j < i; j++)
            {
                offsets[j] += loopVars[j];
            }
        }

        var letBuilders = Enumerable.Range(0, value.DimNames.Length).Select(i => new List<ISequentialBuilder<Expr>>()).ToArray();

        foreach (var (bid, bufferInfo) in nodeMemo.BufferInfoMap)
        {
            if (nodeMemo.DefUseMap.ContainsKey(bid))
            {
                continue;
            }

            for (int i = 0; i < value.DimNames.Length; i++)
            {
                var place = bufferInfo.Places[i];
                for (int sl = 0; sl < place.Length; sl++)
                {
                    if (_sol.Value(place[sl]) == 1)
                    {
                        var subViewInfo = AllocateSubView(value, bid, bufferInfo.Map, forwardOffsets[i], bufferInfo.Shapes[i]);
                        var letBuilder = T.Let(out var letVar, subViewInfo.Buffer, $"{bid}_L{value.Level}");
                        letBuilders[i].Add(letBuilder);
                        if (!_subViewMemo.TryGetValue(value, out var subviewMap))
                        {
                            subviewMap = new();
                            _subViewMemo.Add(value, subviewMap);
                        }

                        subviewMap[bid] = new(letVar, subViewInfo.Offsets);
                    }
                }
            }
        }

        for (int i = value.DimNames.Length - 1; i >= 0; i--)
        {
            var cntBuilder = i == 0 ? parentbuilder : loopBuilders[i - 1];
            for (int j = 0; j < letBuilders[i].Count; j++)
            {
                cntBuilder.Body(letBuilders[i][j]);
                cntBuilder = letBuilders[i][j];
            }

            cntBuilder.Body(loopBuilders[i]);
        }

        value.Child.Accept(this, new(loopBuilders[^1], forwardOffsets[0]));

        return default;
    }

    public Unit Visit(OpNode value, Context context)
    {
        var (parentbuilder, partentOffsets) = context;

        var buffers = new Expr[value.BufferShapes.Length];
        for (int i = 0; i < value.BufferShapes.Length; i++)
        {
            var bid = new BufferIdenitity(value, i);
            var subViewInfo = AllocateSubView(value, bid, OpNodeMemo[value].Maps[i], partentOffsets, OpNodeMemo[value].Shapes[i]);

            buffers[i] = subViewInfo.Buffer;
        }

        parentbuilder.Body(new Call(value.Op, buffers));

        return default;
    }

    private TIR.Buffer AllocateBuffer(BufferIdenitity bid)
    {
        var expr = bid.Node.Buffers[bid.Index];
        if (!BufferMemo.TryGetValue(bid, out var buffer))
        {
            if (expr is IR.Buffers.BufferOf bufof)
            {
                var ttype = bufof.Input.CheckedType switch
                {
                    TensorType t => t,
                    DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt),
                    _ => throw new NotSupportedException(),
                };

                buffer = T.AttachBuffer(None.Default, ttype, MemoryLocation.Input, 1, out _, $"{bid}");
            }
            else if (expr is Call { Target: IR.Buffers.Uninitialized } c)
            {
                var ttype = c.CheckedType switch
                {
                    TensorType t => t,
                    DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt),
                    _ => throw new NotSupportedException(),
                };

                buffer = T.AttachBuffer(None.Default, ttype, MemoryLocation.Data, 1, out _, $"{bid}");
            }
            else
            {
                throw new NotSupportedException();
            }

            BufferMemo.Add(bid, buffer);
        }

        return buffer;
    }

    private SubViewInfo AllocateSubView(ITileAbleNode node, BufferIdenitity bid, AffineMap map, Expr[] forwardOffsets, IntExpr[] shapeExprs)
    {
        var offset = new IR.Tuple(map.Apply(forwardOffsets, Enumerable.Repeat<Expr>(0, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray());
        var shape = new IR.Tuple(shapeExprs.Select(s => (Expr)_sol.Value(s.Var())).ToArray());
        Expr parentBuffer = null!;
        IR.Tuple parentOffsets = null!;

        var parentNode = node.GetParentTileableNode();
        while (parentNode is TileNode parentTileNode)
        {
            if (_subViewMemo.TryGetValue(parentTileNode, out var subViewMap) && subViewMap.TryGetValue(bid, out var subViewInfo))
            {
                parentBuffer = subViewInfo.Buffer;
                parentOffsets = subViewInfo.Offsets;
                break;
            }

            parentNode = parentNode.GetParentTileableNode();
        }

        parentBuffer ??= AllocateBuffer(bid);

        if (parentOffsets is not null)
        {
            var subOffset = new Expr[offset.Count];
            for (int j = 0; j < subOffset.Length; j++)
            {
                var x = offset.Fields[j] - parentOffsets.Fields[j];
                subOffset[j] = CompilerServices.ERewrite(x, new Passes.IRewriteRule[] { new Passes.Rules.Arithmetic.AssociateAdd(), new Passes.Rules.Arithmetic.CommutateAdd(), new Passes.Rules.Arithmetic.XNegX(), new Passes.Rules.Arithmetic.XNegX0() }, new(), CompileOptions);
            }

            offset = new IR.Tuple(subOffset);
        }

        return new SubViewInfo(IR.F.Buffer.BufferSubview(parentBuffer, offset, shape), offset);
    }

    public sealed record Context(ISequentialBuilder<Expr> ParentBuilder, Expr[] ForwardOffsets)
    {
    }

    public sealed record SubViewInfo(Expr Buffer, IR.Tuple Offsets)
    {
    }
}
