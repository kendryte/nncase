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

namespace Nncase.Passes.Rules.ShapeBucket;

public record FusionShapeData(IValue Outshape, IValue[] InputShapes);

public class FusionShapeUpdater : ExprVisitor<Expr, Unit>
{
    private Dictionary<Expr, IValue> Memo;

    public Dictionary<BucketFusion, FusionShapeData> FusionShape = new();

    public FusionShapeUpdater(Dictionary<Expr, IValue> memo)
    {
        Memo = memo;
    }

    protected override Expr DefaultVisitLeaf(Expr expr) => expr;

    protected override Expr VisitLeafCall(Call expr)
    {
        // todo: shapeof的处理
        if (expr.Target is BucketFusion f)
        {
            // 这里算的是value
            var argShape = expr.Arguments.ToArray().Select(arg => GetShape(Memo[arg])).ToArray();
            var shape = GetShape(Memo[expr]);
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
    private DateTime _startTime;
    private string _name;

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
    public RecordFusionShape(Dictionary<BucketFusion, FusionShapeData[]> shapeList)
    {
        FusionShapeInfo = shapeList;
    }

    private Dictionary<Var, int[]> DimVarValues;

    // fusion / info()
    public Dictionary<BucketFusion, FusionShapeData[]> FusionShapeInfo;

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction main, RunPassContext context)
    {
        var t = new SimpleTimer("record fusion shape");
        var options = CompileSession.CompileOptions.ShapeBucketOptions;
        var varMap = options.VarMap;
        DimVarValues = ShapeBucketPassContext.MakeVarValuesForAllSegment(options);
        // 一共有多组key seg
        var list = Enumerable.Range(0, DimVarValues.First().Value.Length).Select(i =>
        {
            // 一组里面多个key seg
            return DimVarValues.Select(pair => (pair.Key, Value: pair.Value[i])).ToArray();
        }).ToArray();
        var tmpFusionShapeList = list.Select((seg, i) =>
            {
                var varValues = seg.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value));
                var exprValues = seg.ToDictionary(pair => (Expr)pair.Key, pair => (IValue)Value.FromTensor(pair.Value));
                var input = MakeDummyInput(varMap, varValues);
                var body = ((Function)main).Body;
                var memo = GetMemo(body, input);
                var f = new FusionShapeUpdater(ConcatDictionary(memo, exprValues));
                f.Visit(main);
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

    private static Dictionary<Expr, IValue> ConcatDictionary(Dictionary<Expr, IValue> memo, Dictionary<Expr, IValue> exprValues)
    {
        foreach (var (key, value) in exprValues)
        {
            if (memo.ContainsKey(key))
            {
                Console.WriteLine();
            }
            memo[key] = value;
        }

        return memo;
    }

    private static Dictionary<Expr, IValue> GetMemo(Expr input, Dictionary<Var, IValue> varValues)
    {
        var visitor = new EvaluateVisitor(varValues, new());
        visitor.Visit(input);
        return visitor.ExprMemo;
    }

    // make dummy value from InputInfo
    // VarInfo:(DimVar -> Value)
    private static Dictionary<Var, IValue>
        MakeDummyInput(IReadOnlyDictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo)
    {
        return info.ToDictionary(
            pair => pair.Key,
            pair =>
            {
                // todo: dummy input可能会有问题...
                var shapeExpr = pair.Key.CheckedShape.IsScalar
                    ? (Expr)Array.Empty<int>()
                    : Stack(new IR.Tuple(pair.Value.Select(x => Cast(x, DataTypes.Int32)).ToArray()), 0);

                var shape = shapeExpr.Evaluate(varInfo).AsTensor();
                return ConstantOfShape(
                    shape,
                    Cast(1, pair.Key.CheckedDataType)).Evaluate(varInfo);
            });
    }
}
