// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using System.Transactions;
using DryIoc;
using DryIoc.ImTools;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Toolkit.HighPerformance;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketHelper;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;
using Dimension = Nncase.IR.Dimension;
using Tuple = System.Tuple;

namespace Nncase.Passes.Rules.ShapeBucket;

public class BucketFusion : Fusion
{
    public BucketFusion(string name, string moduleKind, Expr body, ReadOnlySpan<Var> parameters, Var[] effectVar)
       : base(
            name, moduleKind, body, parameters)
    {
        EffectVar = effectVar;
    }

    public BucketFusion(string moduleKind, Expr body, ReadOnlySpan<Var> parameters, Var[] effectVar)
        : base(
            moduleKind,
            body,
            parameters)
    {
        EffectVar = effectVar;
    }

    public BucketFusion(string name, string moduleKind, Var[] effectVar, Expr body, params Var[] parameters)
        : base(name, moduleKind, body, parameters)
    {
        EffectVar = effectVar;
    }

    public BucketFusion(string moduleKind, Var[] effectVar, Expr body, params Var[] parameters)
        : base(moduleKind, body, parameters)
    {
        EffectVar = effectVar;
    }

    public Var[] EffectVar { get; set; }

    public bool IsSimple
    {
        get
        {
            var names = Name.Split("_");
            var list = new[] { "MatMul", "Conv2D", "Conv2DTranspose", "Transpose" };
            foreach (string name in names)
            {
                if (list.Contains(name))
                {
                    return false;
                }
            }

            return true;
        }
    }

    public static BucketFusion FromNormalFusion(Fusion f, Var[] effectVars)
    {
        return new BucketFusion(f.Name, "stackvm", f.Body, f.Parameters.ToArray(), effectVars);
    }

    public new BucketFusion With(string? name = null, string? moduleKind = null, Expr? body = null, Var[]? parameters = null)
        => new BucketFusion(name ?? Name, moduleKind ?? ModuleKind, body ?? Body, parameters ?? Parameters, EffectVar);
}

[RuleGenerator]
public partial class CallToFusion : RewriteRule<Pattern>
{
    public static int _counter;

    private Call? _currentCall;

    public string ModuleKind => "stackvm";

    public override Pattern Pattern => throw new InvalidOperationException();

    protected virtual bool MustHaveMarker => true;

    private string Name => _currentCall!.Target.GetType().Name;

    private string RelPath => $"{_counter}_{_currentCall!.Target.GetType().Name}";

    protected virtual Expr ProcessForNewBody(Expr expr) => expr;

    protected virtual Expr ProcessForOuterCall(Expr expr) => expr;

    protected virtual (Expr, int)[] CollectInputs(Call call) =>
        call.Arguments.ToArray().Select((arg, i) =>
        {
            if (arg is Marker m && m.Target is not TensorConst)
            {
                return (m, i);
            }

            return (arg, -1);
        }).Where(pair => pair.Item2 != -1).Select(pair => (pair.arg, pair.Item2)).ToArray();

    public virtual bool Check(Call call)
    {
        return true;
    }

    protected virtual void Init(IMatchResult result) { }

    public Expr? GetReplace(Call call, IMatchResult matchResult)
    {
        var originType = call.CheckedType;
        _currentCall = call;
        DumpIR((Expr)matchResult.Root, "origin", RelPath);
        if (!Check(call))
        {
            return null;
        }
        Init(matchResult);

        Console.WriteLine(call.Target.GetType().Name);
        var argsMarkerData = CollectInputs(call);
        var args = argsMarkerData.Select(pair => pair.Item1).ToArray();
        // var argsMarker = argsMarkerData.Select(pair => pair.Item1).ToArray();
        // var args = argsMarker.Select(arg => arg.Target).ToArray();
        var varMap = CompileSession.CompileOptions.ShapeBucketOptions.VarMap;
        var set = MakeEffectVarArray(CompileSession, varMap, args);
        var fusionVars = MakeNewParam(args);
        var newCall = MakeNewCall(call, fusionVars, argsMarkerData);
        var f = MakeNewFusion(fusionVars, args, newCall, set);
        var outerCall = MakeNewOuterCall(newCall, f, args);
        DumpIR(outerCall, "after", RelPath);
        ArgsChecker(args);
        _counter++;

        if (!outerCall.InferenceType())
        {
            DumpIR(outerCall, "InvalidType");
            throw new InvalidOperationException();
        }

        if (outerCall.CheckedType != originType)
        {
            DumpIR(outerCall, "TypeChanged");
            throw new InvalidOperationException();
        }
        return outerCall;
    }

    private static Var[] MakeNewParam(Expr[] args)
    {
        var fusionVars = args.Select(arg => new Var(arg.CheckedType)).ToArray();
        return fusionVars;
    }

    private Expr MakeNewOuterCall(Expr call, BucketFusion f, Expr[] argsMarker)
    {
        // PrintEffectVar(f.Name, set);
        Expr outerCall = ProcessForOuterCall(new Call(f, argsMarker));
        return outerCall;
    }

    private BucketFusion MakeNewFusion(Var[] fusionVars, Expr[] args, Expr newCall, Var[] set)
    {
        // 处理其他的参数用到了分段的input的情况
        // 即便body只有一个call,但这里是针对所有参数的表达式进行替换，比如反卷积的output shape是一个用到了需要分段的input的表达式
        // 如果不加这个则output shape引用的原始的未分段的输入会再次塞进来

        // todo: 如果其中一个arg有多个user,并且有在fusion之外的部分，如果被替换为var,那么fusion外的那部分表达式operand都会跟着变成var
        // args是call的话重新构造就好了, 能够解决目前的情况，但是不知道更复杂的情况会不会出问题

        var body = fusionVars.Zip(args).Aggregate(newCall, (newBody, tuple) =>
        {
            var (fusionVar, arg) = tuple;
            return ReplaceUtility.ReplaceExpr(newBody, arg, fusionVar);
        });

        var f = new BucketFusion($"{Name}_{_counter}", ModuleKind, set, body, fusionVars);
        return f;
    }

    private Expr MakeNewCall(Call call, Var[] fusionVars, (Expr, int)[] argsMarkerData)
    {
        var inputsWithMarkerAndIndex =
            fusionVars.Zip(argsMarkerData).Select(pair =>
            {
                var (arg, originIndex) = pair.Second;
                if (arg is Marker m)
                {
                    return (originIndex, m.With(target: pair.First));
                }

                return (originIndex, arg);
            }).ToArray();

        // index should map to origin input, not inputsWithMarker index
        // var pairs = inputsWithMarkerAndIndex.Select((input, i) => (i, (Expr)input)).ToArray();

        var indices = inputsWithMarkerAndIndex.Select(x => x.originIndex).ToArray();
        var newArgs = call.Arguments.ToArray().Select((arg, i) =>
        {
            if (indices.Contains(i))
            {
                var fields = inputsWithMarkerAndIndex.Where(x => x.originIndex == i).ToArray();

                // todo: tuple type(split) maybe error
                if (arg is IR.Tuple tup)
                {
                    // 包含tuple中所有元素，const以及非const
                    var newFields = new List<Expr>();
                    int inputCounter = 0;
                    foreach (var inputField in tup.Fields.ToArray())
                    {
                        if (inputField is TensorConst)
                        {
                            newFields.Add(inputField);
                        }
                        else
                        {
                            newFields.Add(fields[inputCounter++].arg);
                        }
                    }
                    // var newFields = inputsWithMarkerAndIndex.Select(x => x.originIndex == i ? x.arg : arg).ToArray();
                    return new IR.Tuple(newFields.ToArray());
                }
                if (fields.Length > 1)
                {
                    throw new InvalidOperationException();
                }

                return fields.First().arg;
            }

            return arg;
        }).ToArray();
        // arguments用到其他input的地方就要replace对应的input
        var newCall = call.With(arguments: newArgs);
        // var newCall = ReplaceUtility.ReplaceCallParams(call.Target, call.Arguments.ToArray(), inputsWithMarkerAndIndex);
        var newCallWithMarker = ProcessForOuterCall(newCall);
        return newCallWithMarker;
    }
}

public class MarkerCallToFusion<T> : CallToFusion
    where T : Op
{
    public override Pattern Pattern => IsRangeOfMarker(
        "callMarker",
        IsCallWildcard("call", IsOp<T>()),
        IsTensorConst());

    private Marker callMarker;

    protected override Expr ProcessForNewBody(Expr expr) => callMarker.With(target: expr);

    protected override Expr ProcessForOuterCall(Expr expr) => callMarker.With(target: expr);

    protected override void Init(IMatchResult result)
    {
        callMarker = (Marker)result["callMarker"];
    }
}


public class MultiUserCallToFusion : CallToFusion
{
    private Expr body;
    public override Pattern Pattern => IsWildcard("call", expr =>
    {
        if (expr is Call c && c.Target is not BucketFusion)
        {
            if (c.Target is IR.Tensors.Reshape)
            {
                if (c.Arguments[IR.Tensors.Reshape.Shape.Index] is TensorConst)
                {
                    return CallValidator.ValidTarget(c.Target);
                }
            }
            else
            {
                return CallValidator.ValidTarget(c.Target);
            }
        }

        return false;
    });

    protected override (Expr, int)[] CollectInputs(Call call) =>
        call.Arguments.ToArray().SelectMany((arg, i) =>
        {
            if (arg is IR.Tuple tuple)
            {
                return tuple.Fields
                    .ToArray()
                    .Where(field => field is not TensorConst)
                    .Select(field => (field, i))
                    .ToArray();
            }

            if (arg is not TensorConst)
            {
                return new[] { (arg, i) };
            }

            return new[] { (arg, -1) };
        }).Where(pair => pair.Item2 != -1).Select(pair => (pair.Item1, pair.Item2)).ToArray();

    public override bool Check(Call call)
    {
        return !call.Users.ToArray().OfType<Var>().Any();
        // call.Users.ToArray().OfType<Call>()
        // var names = string.Join("\n", call.Users.ToArray().OfType<Call>().Select(c => ((BucketFusion)c.Target).Name).ToArray());
        // Console.WriteLine(names);
        // Console.WriteLine();

        // if (!call.Users.Where(x => x != body).All(user => user is Call c && c.Target is BucketFusion))
        // {
            // return false;
            throw new NotImplementedException();
        // }

        return true;
    }
}

public class Conv2DToFusion : MarkerCallToFusion<Conv2D>
{
}

public class Conv2DTransposeToFusion : MarkerCallToFusion<Conv2DTranspose>
{
    // when OutputShape is Const, it means output shape is not effected by input.
    public override bool Check(Call call) => call.Arguments[Conv2DTranspose.OutputShape.Index] is not Const;
}

public class MatmulToFusion : MarkerCallToFusion<MatMul>
{
}

public class ActToFusion : MarkerCallToFusion<ActivationOp>
{

}

public class SigmoidToFusion : MarkerCallToFusion<Sigmoid>
{
}

public class LeakyReluToFusion : MarkerCallToFusion<LeakyRelu>
{
}

public class ReluToFusion : MarkerCallToFusion<Relu>
{
}

public class TransposeToFusion : MarkerCallToFusion<Transpose>
{
    protected override bool MustHaveMarker => false;
}

public class PadToFusion : MarkerCallToFusion<Pad>
{
    protected override bool MustHaveMarker => false;

    public override bool Check(Call call) => ((Pad)call.Target).PadMode == PadMode.Constant;
}

public class UnaryToFusion : MarkerCallToFusion<Unary>
{
    public override bool Check(Call call)
    {
        var list = new[] { UnaryOp.Abs, UnaryOp.Neg, UnaryOp.Acos, UnaryOp.Asin };
        var op = ((Unary)call.Target).UnaryOp;
        return call.CheckedShape.Rank > 1 && list.Contains(op);
    }
}

// todo: do more check for binary
public class BinaryToFusion : MarkerCallToFusion<Binary>
{
    // public override bool Check(Call call) => call.CheckedShape.Rank > 1;
}

[RuleGenerator]
public partial class ClearRequire : RewriteRule<Pattern>
{
    // for require(true, value, msg)
    public override Pattern Pattern { get; } = IsRequire(require => true, IsTensorConst("predicate"), IsWildcard("expr"));

    public Expr? GetReplace(bool predicate, Expr expr)
    {
        if (predicate)
        {
            return expr;
        }

        return null;
    }
}

[RuleGenerator]
public partial class FoldRepeatMarker : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = IsRangeOfMarker(
        "markerA",
        IsRangeOfMarker(
            "markerB",
            IsWildcard(),
            IsWildcard("rangeB")),
        IsWildcard("rangeA"));

    public Expr? GetReplace(Expr rangeA, Expr rangeB, Marker markerB)
    {
        if (rangeA == rangeB)
        {
            return markerB;
        }

        return null;
    }
}

[RuleGenerator]
public partial class ClearFusionOuterMarker : RewriteRule<Pattern>
{
    public static Pattern CallerPattern => IsCall(
        "caller",
        IsFusion(null, "stackvm", IsWildcard(), GenerateParameters(null)),
        GenerateParameters(null));

    public override Pattern Pattern { get; } = IsRangeOfMarker("marker", CallerPattern, IsWildcard());

    public Expr? GetReplace(Marker marker, Call caller)
    {
        return caller;
    }
}

[RuleGenerator]
public partial class FusionBucket : RewriteRule<Pattern>
{
    private static int _counter;

    private static string _relPath = string.Empty;

    public override Pattern Pattern => IsCall(
        "outerCall",
        IsFusion(
            "fusion",
            "stackvm",
            IsWildcard("fusionBody"),
            GenerateParameters(null)),
        GenerateParameters(null));

    public static int[] ComputeSegmentList(int segmentCount, int min, int max)
    {
        var size = (max - min) / segmentCount;
        return Enumerable.Range(0, segmentCount - 1).Select(i => min + (i * size)).Append(max).ToArray();
    }

    public static Expr PreProcess(Var input, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, IValue> varValues, Dictionary<Var, Expr[]> fusionInputData, Expr[] fusionInputs, int i)
    {
        var fixedShape = ShapeEvaluate(input, inputInfo, varValues, fusionInputData);
        return new Call(new BucketPad(), input, fixedShape);
    }

    // info:(InputVar -> DimVar)
    // VarInfo:(DimVar -> Value)
    // fusionInfo:(InputVar -> DimVar)
    public static int[] ShapeEvaluate(Expr expr, Dictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo, Dictionary<Var, Expr[]> fusionInfo)
    {
        // var info is used for compute shape expr
        var dummyInput = MakeDummyInput(info, varInfo);
        var fusionDummyInput =
            MakeDummyInput(
                fusionInfo,
                varInfo.Concat(dummyInput).ToDictionary(pair => pair.Key, pair => pair.Value));
        var shapeExpr =
            expr.EvaluateShapeExpr(info.Concat(fusionInfo).ToDictionary(pair => pair.Key, pair => pair.Value));

        if (!shapeExpr.InferenceType())
        {
            throw new InvalidOperationException();
        }

        // used for shape expr evaluate
        // 1. main input
        // 2. fusion input
        // 3. shape var
        var newEvaluatorInfo = dummyInput.Concat(fusionDummyInput).Concat(varInfo)
            .ToDictionary(pair => pair.Key, pair => pair.Value);

        var shape = shapeExpr.Evaluate(newEvaluatorInfo);
        return shape.AsTensor().ToArray<int>();
    }

    public static (Dictionary<Var, IValue> MinDict, Dictionary<Var, IValue> MaxDict) GetBoundDict(
        Dictionary<Var, Expr[]> inputInfo, ShapeBucketOptions options)
    {
        // find vars in Input ShapeExpr
        var vars = inputInfo.Values.SelectMany(x => x).OfType<Var>().ToHashSet().ToArray();

        // DimVarName -> Dict.key -> Dict.Value
        var minDict = options.RangeInfo.ToDictionary(
            pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => (IValue)Value.FromTensor(pair.Value.Min));
        var maxDict = options.RangeInfo.ToDictionary(
            pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => (IValue)Value.FromTensor(pair.Value.Max));
        return (minDict, maxDict);
    }

    public static Expr MakeSplitEntry(Expr originBody, Var[] fusionVars, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, IValue> varInfo, Dictionary<Var, Expr[]> fusionInputdata, Dictionary<Var, Expr[]> fusionInputsShape, Expr[] fusionInputs, string relPath, int seg)
    {
        // 避免这里的修改影响到原始的body，每个分支需要进行自己的修改
        var call = originBody.Clone();

        // 找到拷贝的call里面所有var，和fusion的原始var要对应上
        var finder = new FindVar();
        finder.Visit(call);
        var newVars = finder.Vars;

        var fixInputs = fusionVars
            .Select((arg, i) => PreProcess(arg, inputInfo, varInfo, fusionInputdata, fusionInputs, i)).ToArray();

        // 替换逻辑：新的body中的var -> fusion原始的var -> target为fusion的call的input
        // 本质上只是对这个body的所有输入做替换
        call = fusionVars.Select(v => newVars.FindFirst(newVar => newVar.Name == v.Name)).Zip(fixInputs).Aggregate(
            call,
            (sum, pair) =>
            {
                return ReplaceExpr(sum, pair.First, pair.Second);
            });
        if (!call.InferenceType())
        {
            DumpIR(call, "InvalidType");
            throw new InvalidOperationException();
        }

        if (call.CheckedType is TupleType tuple)
        {
            var fields = Enumerable.Range(0, tuple.Count)
                .Select(i => MakeSlice(originBody[i], fusionInputsShape, call[i])).ToArray();
            return new IR.Tuple(fields);
        }
        else
        {
            return MakeSlice(originBody, fusionInputsShape, call);
        }
    }

    private static Expr MakeSlice(Expr originBody, Dictionary<Var, Expr[]> fusionInputsShape, Expr call)
    {
        var originShape = originBody.EvaluateShapeExpr(fusionInputsShape);
        originShape.InferenceType();
        DumpIR(originShape, "OriginShapeExpr", _relPath);
        var rank = call.CheckedShape.Rank;

        // 对body的输出进行slice
        var body = (Expr)Slice(call, Enumerable.Repeat(0, rank).ToArray(), Cast(originShape, DataTypes.Int32), rank);
        return body;
    }

    public Expr FixInput(Expr body, int[][] shapeList, Var[] fusionVars, Expr[] outerArgs)
    {
        var result = fusionVars.Zip(outerArgs).Zip(shapeList).Aggregate(body, (sum, data) =>
        {
            var ((fusionVar, arg), fixShape) = data;
            Expr expr = new Call(new FixShape(), arg, fixShape);
            if (arg is Marker m)
            {
                expr = m.With(target: expr);
            }

            return ReplaceExpr(sum, fusionVar, expr);
        });
        return result;
    }

    public Expr Rebuild(Call outerCall, BucketFusion fusion, Expr fusionBody)
    {
        var newBody = CompilerServices.Rewrite(fusionBody,
            new IRewriteRule[]
            {
                new MatmulToFusion(),
                new Conv2DToFusion(),
                new Conv2DTransposeToFusion(),
                new TransposeToFusion(),
            },
            new());
        return RestoreBodyWithArgs(outerCall.Arguments.ToArray(), fusion.Parameters.ToArray(), newBody);
    }

    public Expr? GetReplace(Call outerCall, BucketFusion fusion, Expr fusionBody)
    {
        if (ShouldRestore(outerCall, fusion))
        {
            return RestoreBodyWithArgs(outerCall.Arguments.ToArray(), fusion.Parameters.ToArray(), fusion.Body);
        }

        Console.WriteLine($"FusionBucketGetReplace {_counter} {fusion.Name}");
        _relPath = $"{_counter}";
        DumpIR(outerCall, $"BucketOriginFusion_{fusion.Name}", _relPath);

        var varMap = CompileSession.CompileOptions.ShapeBucketOptions.VarMap;

        var fusionInputsShapeExpr = MakeFusionInputShapeExpr(outerCall, fusion, varMap);
        CheckAlive(fusionInputsShapeExpr);

        var options = CompileSession.CompileOptions.ShapeBucketOptions;
        var (minDict, maxDict) = GetBoundDict(varMap, options);

        var fusionVars = fusion.Parameters.ToArray();

        // compute fixed input Shape
        var minFixedShapeList = ComputeFixedShape(fusionVars, minDict, varMap, fusionInputsShapeExpr);
        var maxFixedShapeList = ComputeFixedShape(fusionVars, maxDict, varMap, fusionInputsShapeExpr);

        // 2. get dim info(inputIndex, (dimIndex, range)
        var counts = ComputeCounts(minFixedShapeList, maxFixedShapeList, out int totalCount);
        if (IsFixed(totalCount, minFixedShapeList, maxFixedShapeList))
        {
            var fix = FixInput(fusionBody, minFixedShapeList, fusionVars, outerCall.Arguments.ToArray());
            DumpIR(fix, "BucketResultFix", _relPath);
            _counter++;
            return fix;
        }

        // todo: process total count, matmul maybe multi count, but other should check this
        if (totalCount > 1)
        {
            // Console.WriteLine($"{fusion.Name} totalCount > 1");
            // return null;
        }

        var args = outerCall.Arguments.ToArray();
        var fusionInputShapes = MakeShapeOfFusionInput(fusion, args);

        var dimVarValues = MakeVarValuesForAllSegment(options);
        var info = ComputeSegmentInfo(counts, options);
        var body = Split(outerCall, fusion, fusionBody, fusionVars, info, dimVarValues, args, varMap, fusionInputsShapeExpr, fusionInputShapes);
        body.InferenceType();

        if (body.Users.Count > 1)
        {
            throw new InvalidOperationException();
        }

        // FixInput Replace Var
        var newBody = ReplaceFusionVarWithCallArgs(fusion, args, body);

        // let bind
        if (newBody is If @if)
        {
            newBody = IR.F.Math.Require(true, @if.With(paramList: args));
        }

        DumpIR(newBody, "BucketResult", _relPath);
        _counter++;
        return newBody;
        // todo :save if shape
    }

    private static bool IsFixed(int totalCount, int[][] minFixedShapeList, int[][] maxFixedShapeList) =>
        totalCount == 0 || (minFixedShapeList[0].SequenceEqual(maxFixedShapeList[0]) &&
                            minFixedShapeList[1].SequenceEqual(maxFixedShapeList[1]));

    private static bool ShouldRestore(Call outerCall, BucketFusion fusion) => fusion.IsSimple || outerCall.CheckedType is TupleType || outerCall.CheckedShape.Rank == 0 || outerCall.Arguments.ToArray().Any(arg => arg.CheckedType is TupleType);

    private static Expr RestoreBodyWithArgs(Expr[] args, Var[] parameters, Expr body) =>
        parameters.ToArray().Zip(args).Aggregate(body, (sum, data) =>
        {
            var (fusionVar, arg) = data;
            return ReplaceExpr(sum, fusionVar, arg);
        });

    private static void PrintMinMaxShape(int[][] minFixedShapeList, int[][] maxFixedShapeList, string relPath)
    {
        string str = string.Empty;
        Console.Write("min ");
        str += "min ";
        foreach (int[] shape in minFixedShapeList)
        {
            var s = DumpUtility.SerializeShape(shape) + " ";
            str += s;
            Console.Write(s);
        }

        Console.Write("max ");
        str += "max ";
        foreach (int[] shape in maxFixedShapeList)
        {
            var s = DumpUtility.SerializeShape(shape) + " ";
            str += s;
            Console.Write(s);
        }
    }

    // 计算出使用哪个位置的input进行分段
    private static SegmentInfo ComputeSegmentInfo(
        (int InputIndex, (int First, (int First, int Second) Second)[] Range)[] counts, ShapeBucketOptions options)
    {
        var (iIndex, dimIndex, (min, max)) = counts.Select(x =>
        {
            Debug.Assert(x.Range.Length <= 2, "x.range.Length <= 2");
            return (inputIndex: x.InputIndex, x.Range[0].First, x.Range[0].Second);
        }).ToArray().First();

        var segments = ComputeSegmentList(options.SegmentsCount, min, max);
        var info = new SegmentInfo(iIndex, dimIndex, segments);
        return info;
    }

    // make dummy value from InputInfo
    // VarInfo:(DimVar -> Value)
    private static Dictionary<Var, IValue>
        MakeDummyInput(Dictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo) =>
        info.ToDictionary(
            pair => pair.Key,
            pair =>
            {
                var shapeExpr = pair.Key.CheckedShape.IsScalar ? (Expr)Array.Empty<int>() : Stack(new IR.Tuple(pair.Value.Select(x => Cast(x, DataTypes.Int32)).ToArray()), 0);

                // DumpIR(shapeExpr, "DummyInputShapeExpr", _relPath);
                var shape = shapeExpr.Evaluate(varInfo).AsTensor();
                return ConstantOfShape(
                    shape,
                    Cast(0, pair.Key.CheckedDataType)).Evaluate(varInfo);
            });

    private static (int InputIndex, (int First, (int First, int Second) Second)[] Range)[] ComputeCounts(
        int[][] minFixedShapeList, int[][] maxFixedShapeList, out int totalCount)
    {
        (int InputIndex, (int First, (int First, int Second) Second)[] Range)[] counts = minFixedShapeList.Zip(maxFixedShapeList).Select((pair, inputIndex) =>
        {
            var (minShape, maxShape) = pair;

            // (range, dimIndex)
            var range = Enumerable.Range(0, minShape.Length).Zip(minShape.Zip(maxShape)).Where(data =>
            {
                var (dimIndex, pair) = data;
                return pair.First != pair.Second;
            }).ToArray();
            return (inputIndex, range);
        }).Where(pair => pair.range.Length > 0).ToArray();
        totalCount = counts.Length;
        return counts;
    }

    private static Expr ReplaceFusionVarWithCallArgs(BucketFusion fusion, Expr[] args, Expr newBody) =>
        fusion.Parameters.ToArray().Zip(args).Aggregate(newBody, (sum, pair) =>
        {
            var (param, arg) = pair;
            var result = ReplaceExpr(sum, param, arg);
            return result;
        });

    // 计算shape，而不是shape表达式
    private static Dictionary<Var, Expr[]> MakeShapeOfFusionInput(BucketFusion fusion, Expr[] args)
    {
        var fusionInputShapes = fusion.Parameters
            .ToArray()
            .Zip(args)
            .ToDictionary(pair => pair.First, pair =>
            {
                var shape = Cast((Expr)ShapeOf(pair.Second), DataTypes.Int32);
                return Enumerable.Range(0, pair.Second.CheckedShape.Rank).Select(i => shape[i]).ToArray();
            });
        return fusionInputShapes;
    }

    private static void CheckAlive(Dictionary<Var, Expr[]> fusionInputInfo)
    {
        foreach (var value in fusionInputInfo.Values)
        {
            foreach (var expr in value)
            {
                if (!expr.IsAlive)
                {
                    throw new NotImplementedException();
                }
            }
        }
    }

    private static Dictionary<Var, Expr[]> MakeFusionInputShapeExpr(Call call, BucketFusion fusion, Dictionary<Var, Expr[]> varMap)
    {
        var data = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray().Select((arg, i) =>
        {
            // DumpIR(arg, "MakeFusionInputShapeExprArg");
            var result = arg.EvaluateShapeExpr(varMap);
            return Enumerable.Range(0, arg.CheckedShape.Rank).Select(i =>
            {
                var res = result[i];
                return res;
            }).ToArray();
        })).Select(pair => new KeyValuePair<Var, Expr[]>(pair.First, pair.Second));
        var fusionInputData = data.ToDictionary(pair => pair.Key, pair => pair.Value);
        return fusionInputData;
    }

    private int[][] ComputeFixedShape(Expr[] fusionVars, Dictionary<Var, IValue> varInfo, Dictionary<Var, Expr[]> varMap, Dictionary<Var, Expr[]> fusionInputData) =>
        fusionVars.Select((arg, i) =>
        {
            var fixedShape = ShapeEvaluate(arg, varMap, varInfo, fusionInputData);
            return fixedShape;
        }).ToArray();

    // 计算每个var在不同的段下的值
    private Dictionary<Var, int[]> MakeVarValuesForAllSegment(ShapeBucketOptions options)
    {
        int segmentCount = options.SegmentsCount;
        var varRange = options.RangeInfo;
        var varMap = options.VarMap;
        var varAndInputAllSegment = varRange.ToDictionary(pair => pair.Key, pair =>
        {
            var (min, max) = pair.Value;
            var segments = ComputeSegmentList(segmentCount, min, max);
            return segments;
        });

        var vars = varMap.Values.SelectMany(x => x).OfType<Var>().ToHashSet().ToArray();

        // DimVarName -> Dict.key -> Dict.Value
        var varValues = varAndInputAllSegment.ToDictionary(
            pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => { return pair.Value.OrderByDescending(x => x).ToArray(); });
        return varValues;
    }

    private Expr Split(Call outerCall, BucketFusion fusion, Expr fusionBody, Var[] fusionVars, SegmentInfo info, Dictionary<Var, int[]> varValues, Expr[] fusionInputs, Dictionary<Var, Expr[]> varMap, Dictionary<Var, Expr[]> fusionInputData, Dictionary<Var, Expr[]> fusionInputsShape)
    {
        Call GetDefault(IRType x)
        {
            if (x is TupleType)
            {
                Console.WriteLine("TupleType");
                Console.WriteLine(x);
            }

            return ConstantOfShape(new[] { 1 }, Cast(0, ((TensorType)x).DType));
        }

        // do with marker
        // 分段是针对input做的，而不是替换了input。
        // arg var -> compute
        // arg var -> bucket -> compute
        // arg -> bucket -> compute
        var (inputIndex, dimIndex, segments) = info;
        var dim = ShapeOf(fusionInputs[inputIndex])[dimIndex];
        var sp = fusionBody.CheckedType switch
        {
            TupleType tuple => new IR.Tuple(tuple.Fields.ToArray()
                .Select(x =>
                {
                    return GetDefault(x);
                }).ToArray()),
            TensorType _ => (Expr)ConstantOfShape(new[] { 1 }, Cast(0, fusionBody.CheckedDataType)),
        };

        // var sp = ConstantOfShape(new[] { 1 }, Cast(0, fusionBody.CheckedDataType));
        int i = 0;
        var varInfo = varValues.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value[i]));
        var entry = MakeSplitEntry(fusionBody, fusionVars, varMap, varInfo, fusionInputData, fusionInputsShape, fusionInputs,
            _relPath, segments[0]);
        if (!entry.CheckedShape.IsFixed)
        {
            return Rebuild(outerCall, fusion, fusionBody);
        }

        var body = segments.OrderByDescending(x => x).Aggregate(
            (Expr)IR.F.Math.Require(false, sp, "input dim large than limit"),
            (sum, seg) =>
            {
                // 根据var，也就是target为这个fusion的call的参数来进行判断落在哪个段
                var cond = dim <= (long)seg;

                // select var value for current segment
                var varInfo = varValues.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value[i]));
                var thenBody = MakeSplitEntry(fusionBody, fusionVars, varMap, varInfo, fusionInputData, fusionInputsShape, fusionInputs, _relPath, seg);
                var elseBody = sum;
                i++;
                var result = new If(cond, thenBody, elseBody);
                return result;
            });

        return body;
    }
}

internal record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);
