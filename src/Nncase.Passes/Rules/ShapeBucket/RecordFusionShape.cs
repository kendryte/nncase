// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Google.OrTools.Algorithms;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.IR;
using static Nncase.IR.F.Tensors;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketHelper;

namespace Nncase.Passes.Rules.ShapeBucket;

public class FusionShapeData
{
    public IValue Outshape;
    public IValue[] InputShapes;

    public FusionShapeData(IValue outshape, IValue[] inputShapes)
    {
        Outshape = outshape;
        InputShapes = inputShapes;
    }
}

public class FusionShapeUpdater : ExprVisitor<Expr, Unit>
{
    private readonly Dictionary<Expr, IValue> _memo;

    public FusionShapeUpdater(Dictionary<Expr, IValue> memo)
    {
        _memo = memo;
    }

    public Dictionary<BucketFusion, FusionShapeData> FusionShape { get; set; } = new();
    public Dictionary<string, FusionShapeData> FusionNameShape { get; set; } = new();

    protected override Expr DefaultVisitLeaf(Expr expr) => expr;

    protected override Expr VisitLeafCall(Call expr)
    {
        // if (expr.Target is IR.Math.Require require)
        // {
        //     var msg = require.Message;
        //     if (msg.Contains("_input_", StringComparison.Ordinal))
        //     {
        //         var name = msg.Split("_input_")[0];
        //         var index = int.Parse(msg.Split("_input_")[1]);
        //         FusionNameShape[name].InputShapes[index] = GetShape(_memo[expr]);
        //     }
        //     else
        //     {
        //         FusionNameShape[msg].Outshape = GetShape(_memo[expr]);
        //     }
        // }

        if (expr.Target is BucketFusion f)
        {
            var argShape = expr.Arguments.ToArray().Select(arg => GetShape(_memo[arg])).ToArray();
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

public class SimpleTimer : IDisposable
{
    private readonly DateTime _startTime;
    private readonly string _name;

    public SimpleTimer(string name)
    {
        _startTime = System.DateTime.Now;
        _name = name;
    }

    public void Dispose()
    {
        var endTime = System.DateTime.Now;
        var time = endTime - _startTime;
        Console.WriteLine($"{_name} tooks {time.Seconds}");
    }
}

public class RecordFusionShape : FunctionPass
{
    private Dictionary<Var, int[]> _dimVarValues = new();

    public FusionShapeData[] OutShapeList = Array.Empty<FusionShapeData>();

    public RecordFusionShape(Dictionary<BucketFusion, FusionShapeData[]> shapeList)
    {
        FusionShapeInfo = shapeList;
    }

    public Dictionary<BucketFusion, FusionShapeData[]> FusionShapeInfo { get; set; }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction main, RunPassContext context)
    {
        var options = CompileSession.CompileOptions.ShapeBucketOptions;
        var varMap = options.VarMap;
        _dimVarValues = MakeVarValuesForAllSegment(options);

        // 一共有多组key seg
        var list = Enumerable.Range(0, _dimVarValues.First().Value.Length).Select(i =>
        {
            // 一组里面多个key seg
            return _dimVarValues.Select(pair => (pair.Key, Value: pair.Value[i])).ToArray();
        }).ToArray();

        // 算出输入的大致规模，如果太大就不能并行，否则可以，但是要考虑到内存的限制，目前只有melgan需要这样特殊处理
        var body = ((Function)main).Body;
        var tmpFusionShapeList = list.Select((seg, i) =>
            {
                Console.WriteLine("RunStart");
                // GC.Collect();
                // GC.WaitForPendingFinalizers();
                Console.WriteLine("AfterGC");
                var varValues = seg.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value));
                var exprValues = seg.ToDictionary(pair => (Expr)pair.Key, pair => (IValue)Value.FromTensor(pair.Value));
                var input = MakeDummyInput(varMap, varValues);
                Console.WriteLine("Dummy");
                var memo = EvaluatorUtil.GetMemo(body, ConcatDictionary(input, varValues));
                Console.WriteLine("memo");
                var f = new FusionShapeUpdater(ConcatDictionary(memo, exprValues));
                Console.WriteLine("fusion");
                f.Visit(main);
                Console.WriteLine("end");
                return f.FusionShape;
            }).SelectMany(x => x)
            .ToLookup(x => x.Key, x => x.Value)
            .ToDictionary(pair => pair.Key, pair => pair.ToArray());

        GC.Collect();
        foreach (var (f, shapeInfo) in tmpFusionShapeList)
        {
            FusionShapeInfo[f] = shapeInfo;
        }

        return Task.FromResult(main);
    }

    // make dummy value from InputInfo
    // VarInfo:(DimVar -> Value)
    public static Dictionary<Var, IValue>
        MakeDummyInput(IReadOnlyDictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo)
    {
        return info.ToDictionary(
            pair => pair.Key,
            pair =>
            {
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
}
