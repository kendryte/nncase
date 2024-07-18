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
    private readonly ArgumentsInfo _argumentsInfo;
    private readonly Dictionary<ITileAbleNode, Dictionary<BufferIdenitity, SubViewInfo>> _subViewMemo;

    public TreeSolverResultConstructor(Assignment solution, ArgumentsInfo argumentsInfo, Solver solver, IntExpr one, IntExpr zero, IntExpr elem, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo> levelBufferInfos, Dictionary<ITileAbleNode, DomainInfo> domainInfos, CompileOptions compileOptions)
        : base(solver, one, zero, elem, primitiveBufferInfo, levelBufferInfos, domainInfos, compileOptions.TargetOptions)
    {
        _sol = solution;
        _argumentsInfo = argumentsInfo;
        OutSideBufferMemo = new();
        _subViewMemo = new();
        CompileOptions = compileOptions;
    }

    public Dictionary<BufferIdenitity, TIR.Buffer> OutSideBufferMemo { get; }

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

        // forwardOffsets[0] means partentOffsets, forwardOffsets[i] means partentOffsets[0:i] + loop vars[0:i]
        var forwardOffsets = new Expr[loopVars.Length + 1][];
        for (int i = 0; i < loopVars.Length + 1; i++)
        {
            var offsets = forwardOffsets[i] = initOffsets.ToArray();

            for (int j = 0; j < i; j++)
            {
                offsets[j] += loopVars[j];
            }
        }

        // var domainLetBuilders = Enumerable.Range(0, value.DimNames.Length).Select(i => new List<ISequentialBuilder<Expr>>()).ToArray();
        var cntBuilder = parentbuilder;
        for (int i = 0; i < value.DimNames.Length; i++)
        {
            foreach (var (bid, bufferInfo) in nodeMemo.BufferInfoMap)
            {
                var place = bufferInfo.Places[i];
                for (int sl = 0; sl < place.Length; sl++)
                {
                    if (_sol.Value(place[sl]) == 1)
                    {
                        var viewInfo = GetParentSubViewInfo(value, bid, bufferInfo.Map, forwardOffsets[i], bufferInfo.Shapes[i]);
                        var subView = IR.F.Buffer.BufferSubview(viewInfo.Buffer, viewInfo.Offsets, new IR.Tuple(viewInfo.Shape.Select(x => (Expr)x).ToArray()));
                        Var subBufVar;

                        // the parent buffer is temp buffer.
                        if (viewInfo.AllocateBuilder is ISequentialBuilder<Let> allocateBuilder)
                        {
                            var letBuilder = T.Let(out subBufVar, IR.F.Buffer.AllocateBufferView(viewInfo.Buffer), $"{bid}_L{value.Level}");
                            allocateBuilder.Body(letBuilder);
                            cntBuilder.Body(allocateBuilder);
                            cntBuilder = letBuilder;
                        }
                        else
                        {
                            // copy sub view to new created local buffer.
                            // var dtype = viewInfo.Buffer.CheckedDataType;
                            // var spanBuilder = T.Let(out var subBufSpan, IR.F.Buffer.Allocate(TensorUtilities.GetProduct(viewInfo.Shape, 0), dtype, MemoryLocation.L2Data, false), $"{bid}_L{value.Level}_span");
                            // T.AttachBuffer(subBufSpan, new TensorType(dtype, viewInfo.Shape), MemoryLocation.L2Data, 1, out var subBuf, $"{bid}_L{value.Level}")
                            var letBuilder = T.Let(out subBufVar, subView, $"{bid}_L{value.Level}");
                            cntBuilder.Body(letBuilder);
                            cntBuilder = letBuilder;
                        }

                        if (!_subViewMemo.TryGetValue(value, out var subViewMap))
                        {
                            subViewMap = new();
                            _subViewMemo.Add(value, subViewMap);
                        }

                        subViewMap[bid] = new(subBufVar, viewInfo.Offsets);
                    }
                }
            }

            cntBuilder.Body(loopBuilders[i]);
            cntBuilder = loopBuilders[i];
        }

        value.Child.Accept(this, new(loopBuilders[^1], forwardOffsets[^1]));

        return default;
    }

    public Unit Visit(OpNode value, Context context)
    {
        var (parentbuilder, partentOffsets) = context;

        var buffers = new Expr[value.BufferShapes.Length];
        for (int i = 0; i < value.BufferShapes.Length; i++)
        {
            var bid = new BufferIdenitity(value, i);
            var viewInfo = GetParentSubViewInfo(value, bid, OpNodeMemo[value].Maps[i], partentOffsets, OpNodeMemo[value].Shapes[i]);

            buffers[i] = IR.F.Buffer.BufferSubview(viewInfo.Buffer, viewInfo.Offsets, new IR.Tuple(viewInfo.Shape.Select(x => (Expr)x).ToArray()));
        }

        var bodyVarReplaces = new Dictionary<Expr, Expr>();
        for (int i = 0; i < value.Grid.BodyParameters.Length; i++)
        {
            bodyVarReplaces.Add(value.Grid.BodyParameters[i], buffers[i]);
        }

        var domain = new IR.Tuple(partentOffsets.Select(off => new IR.Tuple(off, 0L)).ToArray());
        bodyVarReplaces.Add(value.Grid.DomainParameter, domain);
        var nestBody = new ReplacingExprCloner(bodyVarReplaces).Clone(value.Grid.Body, default);
        parentbuilder.Body(nestBody);

        return default;
    }

    private TensorType GetBufferTensorType(Expr expr)
    {
        TensorType GetTensorType(IRType type) => type switch
        {
            TensorType t => t,
            DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt),
            _ => throw new NotSupportedException(),
        };

        return expr switch
        {
            IR.Buffers.BufferOf bufof => GetTensorType(bufof.Input.CheckedType),
            Call { Target: IR.Buffers.Uninitialized } c => GetTensorType(c.CheckedType),
            _ => throw new NotSupportedException(),
        };
    }

    /// <summary>
    /// declare the input/output buffer.
    /// </summary>
    private TIR.Buffer GetDeclareBuffer(BufferIdenitity bid)
    {
        var expr = bid.Node.Buffers[bid.Index];
        var tensorType = GetBufferTensorType(expr);
        if (!OutSideBufferMemo.TryGetValue(bid, out var buffer))
        {
            buffer = T.AttachBuffer(None.Default, tensorType, MemoryLocation.Data, 1, out _, $"{bid}");

            OutSideBufferMemo.Add(bid, buffer);
        }

        return buffer;
    }

    private bool TryGetParerntBuffer(ITileAbleNode node, BufferIdenitity bid, out Expr parentBuffer, out IR.Tuple parentOffsets)
    {
        var cbid = bid;
        var parentNode = node.GetParentTileableNode();
        while (parentNode is TileNode parentTileNode)
        {
            var pbid = TileNodeMemo[parentTileNode].GetCacheBid(cbid);
            if (_subViewMemo.TryGetValue(parentTileNode, out var subViewMap) && subViewMap.TryGetValue(pbid, out var subViewInfo))
            {
                parentBuffer = subViewInfo.Buffer;
                parentOffsets = subViewInfo.Offsets;
                return true;
            }

            parentNode = parentNode.GetParentTileableNode();
            cbid = pbid;
        }

        parentBuffer = null!;
        parentOffsets = null!;
        return false;
    }

    private ParentSubViewInfo GetParentSubViewInfo(ITileAbleNode node, BufferIdenitity bid, AffineMap map, Expr[] forwardOffsets, IntExpr[] shapeExprs)
    {
        var offset = new IR.Tuple(map.Apply(forwardOffsets, Enumerable.Repeat<Expr>(0, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray());
        var shape = shapeExprs.Select(s => (int)_sol.Value(s.Var())).ToArray();
        ISequentialBuilder<Let>? allocateBuilder = null;
        if (TryGetParerntBuffer(node, bid, out var parentBuffer, out var parentOffsets))
        {
            var subOffset = new Expr[offset.Count];
            for (int j = 0; j < subOffset.Length; j++)
            {
                var x = offset.Fields[j] - parentOffsets.Fields[j];
                subOffset[j] = x;
                // CompilerServices.ERewrite(x, new Passes.IRewriteRule[] { new Passes.Rules.Arithmetic.AssociateAdd(), new Passes.Rules.Arithmetic.CommutateAdd(), new Passes.Rules.Arithmetic.XNegX(), new Passes.Rules.Arithmetic.XNegX0() }, new(), CompileOptions);
            }

            offset = new IR.Tuple(subOffset);
        }
        else
        {
            if (_argumentsInfo.GetBufferKind(bid) is not ArgumentsInfo.BufferKind.None)
            {
                parentBuffer = GetDeclareBuffer(_argumentsInfo.GetUniqueIdenitity(bid));
            }
            else
            {
                parentBuffer = GetAllocateBuffer(bid, out allocateBuilder);
            }
        }

        return new ParentSubViewInfo(parentBuffer, offset, shape, allocateBuilder);
    }

    /// <summary>
    /// Get the local allocate buffer.
    /// </summary>
    private TIR.Buffer GetAllocateBuffer(BufferIdenitity bid, out ISequentialBuilder<Let> scope)
    {
        var expr = bid.Node.Buffers[bid.Index];
        var tensorType = GetBufferTensorType(expr);
        if (!OutSideBufferMemo.TryGetValue(bid, out var buffer))
        {
            var alloc = IR.F.Buffer.Allocate(TensorUtilities.GetProduct(tensorType.Shape.ToValueArray()), tensorType.DType, MemoryLocation.L2Data, false);
            scope = T.Let(out var allocVar, alloc, $"{bid}_span");
            buffer = T.AttachBuffer(allocVar, tensorType, MemoryLocation.L2Data, 1, out _, $"{bid}");
            OutSideBufferMemo.Add(bid, buffer);
        }
        else
        {
            throw new InvalidOperationException("can't allocate twice.");
        }

        return buffer;
    }

    public sealed record Context(ISequentialBuilder<Expr> ParentBuilder, Expr[] ForwardOffsets)
    {
    }

    public sealed record ParentSubViewInfo(Expr Buffer, IR.Tuple Offsets, int[] Shape, ISequentialBuilder<Let>? AllocateBuilder)
    {
    }

    public sealed record SubViewInfo(Expr Buffer, IR.Tuple Offsets)
    {
    }
}
