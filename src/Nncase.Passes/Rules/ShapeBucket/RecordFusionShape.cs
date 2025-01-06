// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Google.OrTools.Algorithms;
using Google.OrTools.Graph;
using Microsoft.Toolkit.HighPerformance;
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
    public FusionShapeData(IValue outshape, IValue[] inputShapes, IValue?[] inputValues, bool[] inputFromShapes)
    {
        Outshape = outshape;
        InputShapes = inputShapes;
        InputValues = inputValues;
        InputFromShapes = inputFromShapes;
    }

    public IValue Outshape { get; }

    public IValue[] InputShapes { get; }

    public IValue?[] InputValues { get; }

    public bool[] InputFromShapes { get; }
}

public class FusionShapeUpdater2 : ExprVisitor<Expr, Unit>
{
    private readonly Dictionary<Expr, ValueOrShape> _memo;

    public FusionShapeUpdater2(Dictionary<Expr, ValueOrShape> memo)
    {
        _memo = memo;
    }

    public Dictionary<BucketFusion, FusionShapeData> FusionShape { get; } = new();

    protected override Expr DefaultVisitLeaf(Expr expr) => expr;

    protected override Expr VisitLeafCall(Call expr)
    {
        if (expr.Target is BucketFusion f)
        {
            var argData = expr.Arguments.ToArray().Select(arg =>
            {
                var exp = arg is Marker m ? m.Target : arg;
                var valueOrShape = _memo[exp];
                return (Shape: GetValueOfShape(valueOrShape.IRType!), Value: valueOrShape.Value, FromShape: valueOrShape.FromShape);
            }).ToArray();
            var shape = GetValueOfShape(_memo[expr].IRType!);
            FusionShape[f] = new FusionShapeData(shape, argData.Select(x => x.Shape).ToArray(), argData.Select(x => x.Value).ToArray(), argData.Select(x => x.FromShape).ToArray());
        }

        return expr;
    }

    private IValue GetValueOfShape(IRType type)
    {
        return type switch
        {
            TensorType t => Value.FromTensor(t.Shape.ToValueArray()),
            TupleType tp => Value.FromTensors(tp.Select(tp => GetValueOfShape(tp).AsTensor()).ToArray()),
            _ => throw new InvalidOperationException(),
        };
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

    public static Dictionary<Var, IRType>
            MakeDummyInputType(IReadOnlyDictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo)
    {
        return info.ToDictionary(
            pair => pair.Key,
            pair =>
            {
                if (pair.Key.CheckedShape.IsFixed)
                {
                    return pair.Key.CheckedType;
                }

                // todo: dummy input可能会有问题...
                var shapeExpr = pair.Key.CheckedShape.IsScalar
                    ? (Expr)Array.Empty<int>()
                    : Stack(new IR.Tuple(pair.Value.Select(x => Cast(x, DataTypes.Int64)).ToArray()), 0);

                var shape = shapeExpr.Evaluate(varInfo).AsTensor();
                return new TensorType(pair.Key.CheckedDataType, new Shape(shape.ToArray<int>()));
            });
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

        var staticShape = IsStaticShpae; // have problem here.
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
        }).Where(kvalues =>
        {
            if (kvalues.Length == 2)
            {
#pragma warning disable SA1008 // Opening parenthesis should be spaced correctly
                (int, int) vv;
                if (kvalues[0].Key.Name == "seq_len" && kvalues[1].Key.Name == "history_len")
                {
                    vv = (kvalues[0].Value, kvalues[1].Value);
                }
                else if (kvalues[1].Key.Name == "seq_len" && kvalues[0].Key.Name == "history_len")
                {
                    vv = (kvalues[1].Value, kvalues[0].Value);
                }
                else
                {
                    return true;
                }

                return vv switch
                {
                    (1, > 0) => true,
                    ( > 0, 0) => true,
                    _ => false,
                };
#pragma warning restore SA1008 // Opening parenthesis should be spaced correctly
            }

            return true;
        }).ToArray();
        var list = _once ? tmpList[^1..] : tmpList;

        var body = ((Function)main).Body;
        var tmpFusionShapeList = list.Select((seg, i) =>
            {
                var varValues = seg.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value));
                var exprValues = seg.ToDictionary(pair => (Expr)pair.Key, pair => (IValue)Value.FromTensor(pair.Value));
#if false
                var input = MakeDummyInput(varMap, varValues);
                var memo = EvaluatorUtil.GetMemo(body, ConcatDictionary(input, varValues));
                var f = new FusionShapeUpdater(ConcatDictionary(memo, exprValues));
#else
                var input = MakeDummyInputType(varMap, varValues);
                var eval = new PartialShapeEvaluator(input.ToDictionary(p => p.Key, p => new ValueOrShape(p.Value, null, false)), varValues);
                eval.Visit(body);
                var memo = eval.ExprMemo;
                foreach (var (k, v) in exprValues)
                {
                    var x = v.AsTensor();
                    memo.Add(k, new(new TensorType(x.ElementType, x.Shape), v, true));
                }

                var f = new FusionShapeUpdater2(memo);
#endif
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

public record ValueOrShape
{
    private IValue? _concreteValue;

    public ValueOrShape(IRType? irType, IValue? value, bool fromShape)
    {
        if (irType is InvalidType)
        {
            throw new InvalidOperationException();
        }

        IRType = irType;
        Value = value;
        _concreteValue = null;
        FromShape = fromShape;
    }

    public IRType? IRType { get; }

    public IValue? Value { get; }

    public bool HasValue => Value != null;

    public bool FromShape { get; }

    public IValue Concrete()
    {
        if (_concreteValue != null)
        {
            return _concreteValue;
        }

        if (Value is IValue value)
        {
            _concreteValue = value;
            return _concreteValue;
        }

        if (IRType is TensorType { Shape: { IsFixed: true } } ttype)
        {
            _concreteValue = new TensorValue(Tensor.FromScalar<int>(0, ttype.Shape.ToValueArray()).CastTo(ttype.DType));
            return _concreteValue;
        }

        throw new NotSupportedException();
    }
}

internal sealed class PartialShapeEvaluator : ExprVisitor<ValueOrShape, Unit>
{
    public PartialShapeEvaluator(Dictionary<Var, ValueOrShape> inputDict, Dictionary<Var, IValue> dimDict)
    {
        FeedDict = inputDict;
        DimDict = dimDict;
    }

    public Dictionary<Var, ValueOrShape> FeedDict { get; }

    public Dictionary<Var, IValue> DimDict { get; }

    protected override ValueOrShape VisitLeafMarker(Marker expr) => Visit(expr.Target);

    protected override ValueOrShape VisitLeafBaseFunction(BaseFunction expr) => new(expr.CheckedType, null, false);

    protected override ValueOrShape VisitLeafOp(Op expr) => new(expr.CheckedType, null, false);

    protected override ValueOrShape VisitLeafVar(Var expr)
    {
        if (FeedDict.TryGetValue(expr, out var value))
        {
            return value;
        }
        else if (DimDict.TryGetValue(expr, out var dimValue) && dimValue is TensorValue dimtv)
        {
            return new(dimtv.Type, dimtv, true);
        }
        else
        {
            throw new NotSupportedException("11");
        }
    }

    protected override ValueOrShape DefaultVisit(Expr expr) => base.DefaultVisit(expr);

    protected override ValueOrShape VisitLeafTuple(IR.Tuple expr)
    {
        var valueOrShapes = expr.Fields.AsValueEnumerable().Select(Visit).ToArray();
        var value = Value.FromTensors(valueOrShapes.Select(vs => vs.Concrete().AsTensor()).ToArray());

        // FIX ME: TupleType's from shape is not correct
        return new(value.Type, value, valueOrShapes.Any(x => x.FromShape));
    }

    protected override ValueOrShape VisitLeafCall(Call expr)
    {
        var args = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        ValueOrShape result;
        switch (expr.Target)
        {
            case IR.Tensors.ShapeOf:
                {
                    var shapeArr = ((TensorType)args[0].IRType!).Shape.Select(x => (long)x.FixedValue).ToArray();
                    var value = Value.FromTensor(Tensor.From<long>(shapeArr));
                    result = new(value.Type, value, true);
                }

                break;
            case Op op:
                {
                    if (args.All(x => x is { HasValue: true }))
                    {
                        var fromShape = args.All(x => x.FromShape);
                        var tmpCall = new Call(op, args.Select(a => Const.FromValue(a.Concrete())).ToArray());
                        var ctx = new EvaluateContext(args)
                        {
                            CurrentCall = tmpCall,
                        };
                        var value = CompilerServices.EvaluateOp(op, ctx);
                        result = new(value.Type, value, fromShape);
                    }
                    else
                    {
                        var ctx = new TypeInferenceContext(args);
                        result = new(CompilerServices.InferenceOp(op, ctx, new()), null, false);
                    }
                }

                break;
            case Fusion fusion:
                {
                    var feedDict = new Dictionary<Var, ValueOrShape>(FeedDict.Concat(fusion.Parameters.ToArray().Zip(args).Select(x => new KeyValuePair<Var, ValueOrShape>(x.First, x.Second))));
                    var eval = new PartialShapeEvaluator(feedDict, DimDict);
                    result = eval.Visit(fusion.Body);
                }

                break;
            case Function func:
                {
                    var feedDict = new Dictionary<Var, ValueOrShape>(FeedDict.Concat(func.Parameters.ToArray().Zip(args).Select(x => new KeyValuePair<Var, ValueOrShape>(x.First, x.Second))));
                    var eval = new PartialShapeEvaluator(feedDict, DimDict);
                    result = eval.Visit(func.Body);
                }

                break;
            default:
                throw new NotSupportedException("fuck!");
        }

        return result;
    }

    protected override ValueOrShape VisitLeafConst(Const expr) => new(expr.CheckedType, Value.FromConst(expr), true);
}

internal sealed class EvaluateContext : IEvaluateContext
{
    public EvaluateContext(ValueOrShape[] args)
    {
        Args = args;
        CurrentCall = null!;
    }

    public ValueOrShape[] Args { get; }

    public BaseCall CurrentCall { get; set; }

    public IValue GetArgumentValue(Op op, ParameterInfo parameter)
    {
        return op.GetType() == parameter.OwnerType
            ? Args[parameter.Index].Value!
            : throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
    }
}

internal sealed class TypeInferenceContext : ITypeInferenceContext
{
    private readonly Expr[] _exprs;

    public TypeInferenceContext(ValueOrShape[] args)
    {
        Args = args;
        _exprs = new Expr[args.Length];
    }

    public ValueOrShape[] Args { get; }

    public Expr GetArgument(Op op, ParameterInfo parameter)
    {
        if (op.GetType() == parameter.OwnerType)
        {
            if (_exprs[parameter.Index] is null)
            {
                _exprs[parameter.Index] = Const.FromValue(Args[parameter.Index].Concrete());
            }

            return _exprs[parameter.Index];
        }
        else
        {
            throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        }
    }

    public Expr[] GetArguments(Op op, params ParameterInfo[] paramsInfo)
    {
        return paramsInfo.Select(info => GetArgument(op, info)).ToArray();
    }

    public IRType GetArgumentType(Op op, ParameterInfo parameter) => Args[parameter.Index].IRType!;
}
