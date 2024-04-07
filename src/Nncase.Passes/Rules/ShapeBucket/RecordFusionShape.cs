// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Google.OrTools.Algorithms;
using Google.OrTools.Graph;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketHelper;

namespace Nncase.Passes.Rules.ShapeBucket;

public class FusionShapeData
{
    public FusionShapeData(IValue outshape, IValue[] inputShapes)
    {
        Outshape = outshape;
        InputShapes = inputShapes;
    }

    public IValue Outshape { get; }

    public IValue[] InputShapes { get; }
}

public class FusionShapeUpdater : ExprVisitor<Expr, Unit>
{
    private readonly Dictionary<Expr, IValue> _memo;

    public FusionShapeUpdater(Dictionary<Expr, IValue> memo)
    {
        _memo = memo;
    }

    public Dictionary<BucketFusion, FusionShapeData> FusionShape { get; } = new();

    protected override Expr DefaultVisitLeaf(Expr expr) => expr;

    protected override Expr VisitLeafCall(Call expr)
    {
        if (expr.Target is BucketFusion f)
        {
            var argShape = expr.Arguments.ToArray().Select(arg =>
            {
                var exp = arg is Marker m ? m.Target : arg;
                return GetShape(_memo[exp]);
            }).ToArray();
            var shape = GetShape(_memo[expr]);
            FusionShape[f] = new FusionShapeData(shape, argShape);
        }

        return expr;
    }

    private IValue GetShape(IValue value)
    {
        var shapes = value.AsTensors().Select(x => x.Shape.ToValueArray()).ToArray();
        if (shapes.Length == 1)
        {
            return Value.FromTensor(shapes[0]);
        }

        return new TupleValue(shapes.Select(x => Value.FromTensor(x)).ToArray());
    }
}

public class RecordFusionShape : FunctionPass
{
    private readonly bool _once;

    private Dictionary<Var, int[]> _dimVarValues = new();

    public RecordFusionShape(Dictionary<BucketFusion, FusionShapeData[]> shapeList, bool once = false)
    {
        FusionShapeInfo = shapeList;
        _once = once;
    }

    public Dictionary<BucketFusion, FusionShapeData[]> FusionShapeInfo { get; set; }

    // make dummy value from InputInfo
    // VarInfo:(DimVar -> Value)
    public static Dictionary<Var, IValue>
        MakeDummyInput(IReadOnlyDictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo)
    {
        return info.ToDictionary(
            pair => pair.Key,
            pair =>
            {
                if (pair.Key.CheckedShape.IsFixed)
                {
                    return ConstantOfShape(
                        pair.Key.CheckedShape.ToValueArray(),
                        Cast(1, pair.Key.CheckedDataType)).Evaluate();
                }

                // todo: dummy input可能会有问题...
                var shapeExpr = pair.Key.CheckedShape.IsScalar
                    ? (Expr)Array.Empty<int>()
                    : Stack(new IR.Tuple(pair.Value.Select(x => Cast(x, DataTypes.Int64)).ToArray()), 0);

                var shape = shapeExpr.Evaluate(varInfo).AsTensor();
                return ConstantOfShape(
                    shape,
                    Cast(1, pair.Key.CheckedDataType)).Evaluate(varInfo);
            });
    }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction main, RunPassContext context)
    {
        var options = CompileSession.CompileOptions.ShapeBucketOptions;
        var varMap = options.VarMap;

        var staticShape = IsStaticShpae;
        var segmentCount = staticShape
                           && SingleDimVar(options)
            ? options.RangeInfo.First().Value.Max
            : options.SegmentsCount;

        _dimVarValues = MakeVarValuesForAllSegment(options, segmentCount, staticShape);

        // 一共有多组key seg
        var tmpList = Enumerable.Range(0, _dimVarValues.First().Value.Length).Select(i =>
        {
            // 一组里面多个key seg
            return _dimVarValues.Select(pair => (pair.Key, Value: pair.Value[i])).ToArray();
        });
        var list = _once ? tmpList.TakeLast(1).ToArray() : tmpList.ToArray();

        var body = ((Function)main).Body;
        var tmpFusionShapeList = list.Select((seg, i) =>
            {
                var varValues = seg.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value));
                var exprValues = seg.ToDictionary(pair => (Expr)pair.Key, pair => (IValue)Value.FromTensor(pair.Value));
                var input = MakeDummyInput(varMap, varValues);
                var memo = EvaluatorUtil.GetMemo(body, ConcatDictionary(input, varValues));
                var f = new FusionShapeUpdater(ConcatDictionary(memo, exprValues));
                f.Visit(main);
                GC.Collect();
                return f.FusionShape;
            }).SelectMany(x => x)
            .ToLookup(x => x.Key, x => x.Value)
            .ToDictionary(pair => pair.Key, pair => pair.ToArray());

        foreach (var (f, shapeInfo) in tmpFusionShapeList)
        {
            FusionShapeInfo[f] = shapeInfo;
        }

        return Task.FromResult(main);
    }
}
