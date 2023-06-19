﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reactive;
using DryIoc.ImTools;
using Microsoft.Toolkit.HighPerformance;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Transforms;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
using static Nncase.Passes.Rules.Neutral.ShapeBucketHelper;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;
using Dimension = Nncase.IR.Dimension;
using Tuple = System.Tuple;

namespace Nncase.Passes.Rules.Neutral;

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

    public new BucketFusion With(string? name = null, string? moduleKind = null, Expr? body = null, Var[]? parameters = null)
        => new BucketFusion(name ?? Name, moduleKind ?? ModuleKind, body ?? Body, parameters ?? Parameters, EffectVar);
}

[RuleGenerator]
public partial class MarkerCallToFusion<T> : RewriteRule<Pattern>
    where T : Op
{
    private Call? _currentCall;

    private int _counter;

    public string ModuleKind => "stackvm";

    public override Pattern Pattern => IsRangeOfMarker(
        "callMarker",
        IsCallWildcard(null, IsOp<T>()),
        IsTensorConst());

    private string Name => _currentCall!.Target.GetType().Name;

    private string RelPath => $"{_counter}_{_currentCall!.Target.GetType().Name}";

    public virtual bool Check(Call call)
    {
        return true;
    }

    public Expr GetReplace(Marker callMarker)
    {
        var call = (Call)callMarker.Target;
        _currentCall = call;
        DumpIR(callMarker, "origin", RelPath);
        var argsMarker = GetCallInputs(call);
        var args = argsMarker.Select(arg => arg.Target).ToArray();
        var varMap = CompileSession.CompileOptions.ShapeBucketOptions.VarMap;
        var set = MakeEffectVarArray(varMap, args);
        var fusionVars = argsMarker.Select(arg => new Var(arg.CheckedType)).ToArray();
        var inputsWithMarker =
            fusionVars.Zip(argsMarker).Select(pair => pair.Second.With(target: pair.First)).ToArray();

        var pairs = inputsWithMarker.Select((input, i) => (i, (Expr)input)).ToArray();

        // arguments用到其他input的地方就要replace对应的input
        var newCall = ReplaceUtility.ReplaceCallParams(call.Target, call.Arguments.ToArray(), pairs);
        var newCallWithMarker = callMarker.With(target: newCall);
        var body = fusionVars.Zip(args).Aggregate((Expr)newCallWithMarker, (newBody, tuple) =>
        {
            var (fusionVar, arg) = tuple;
            return ReplaceUtility.ReplaceExpr(newBody, arg, fusionVar);
        });
        var f = new BucketFusion($"{Name}_{_counter}", ModuleKind, set, body, fusionVars);
        var outerCall = new Call(f, args);
        DumpIR(outerCall, "fusion", RelPath);
        _counter++;
        return outerCall;
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

[RuleGenerator]
public partial class FusionBucket : RewriteRule<Pattern>
{
    private static int _counter;

    private string _relPath = string.Empty;

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
            throw new InvalidOperationException();
        }

        var originShape = originBody.EvaluateShapeExpr(fusionInputsShape);
        originShape.InferenceType();

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
        DumpIR(result, "FixInput", _relPath);
        return result;
    }

    public Expr? GetReplace(Call outerCall, BucketFusion fusion, Expr fusionBody)
    {
        Console.WriteLine($"FusionBucketGetReplace {_counter} {fusion.Name}");
        _relPath = $"{_counter}";
        DumpIR(outerCall, $"BucketOriginFusion_{fusion.Name}", _relPath);

        var varMap = CompileSession.CompileOptions.ShapeBucketOptions.VarMap;

        var fusionInputsShapeExpr = MakeFusionInputShapeExpr(outerCall, fusion, varMap);
        CheckAlive(fusionInputsShapeExpr);

        // ensure alive in rewrite, release when return
        using var pinner = new ExprPinner(fusionInputsShapeExpr.Values.SelectMany(x => x).ToArray());

        var options = CompileSession.CompileOptions.ShapeBucketOptions;
        var (minDict, maxDict) = GetBoundDict(varMap, options);

        var fusionVars = fusion.Parameters.ToArray();

        // compute fixed Shape
        var minFixedShapeList = ComputeFixedShape(fusionVars, minDict, varMap, fusionInputsShapeExpr);
        var maxFixedShapeList = ComputeFixedShape(fusionVars, maxDict, varMap, fusionInputsShapeExpr);
        PrintMinMaxShape(minFixedShapeList, maxFixedShapeList, _relPath);

        // 2. get dim info(inputIndex, (dimIndex, range)
        var counts = ComputeCounts(minFixedShapeList, maxFixedShapeList, out int totalCount);

        if (totalCount == 0 || (minFixedShapeList[0].SequenceEqual(maxFixedShapeList[0]) &&
                                minFixedShapeList[1].SequenceEqual(maxFixedShapeList[1])))
        {
            DumpIR(fusionBody, "BucketResultFix", _relPath);
            var fix = FixInput(fusionBody, minFixedShapeList, fusionVars, outerCall.Arguments.ToArray());
            _counter++;
            return fix;
        }

        var args = outerCall.Arguments.ToArray();
        var fusionInputShapes = MakeShapeOfFusionInput(fusion, args);

        var dimVarValues = MakeVarValuesForAllSegment(options);
        var info = ComputeSegmentInfo(counts, options);
        var body = Split(fusionBody, fusionVars, info, 0, 1, dimVarValues, args, varMap, fusionInputsShapeExpr, fusionInputShapes);
        body.InferenceType();

        // FixInput Replace Var
        var newBody = ReplaceFusionVarWithCallArgs(fusion, args, body);
        DumpIR(newBody, "BucketResult", _relPath);
        _counter++;
        return newBody;
    }

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

        Console.WriteLine();

        // DumpUtility.WriteResult(Path.Join(relPath, "fixShape"), str);
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
                var shapeExpr = Stack(new IR.Tuple(pair.Value.Select(x => Cast(x, DataTypes.Int32)).ToArray()), 0);
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
            return ReplaceExpr(sum, param, arg);
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

    private Expr Split(Expr fusionBody, Var[] fusionVars, SegmentInfo info, int current, int limit, Dictionary<Var, int[]> varValues, Expr[] fusionInputs, Dictionary<Var, Expr[]> varMap, Dictionary<Var, Expr[]> fusionInputData, Dictionary<Var, Expr[]> fusionInputsShape)
    {
        // do with marker
        // 分段是针对input做的，而不是替换了input。
        // arg var -> compute
        // arg var -> bucket -> compute
        // arg -> bucket -> compute
        var (inputIndex, dimIndex, segments) = info;
        var sp = ConstantOfShape(new[] { 1 }, Cast(0, fusionBody.CheckedDataType));
        int i = 0;

        var body = segments.OrderByDescending(x => x).Aggregate(
            (Expr)IR.F.Math.Require(false, sp, "input dim large than limit"),
            (sum, seg) =>
            {
                // 根据var，也就是target为这个fusion的call的参数来进行判断落在哪个段
                var cond = Cast(ShapeOf(fusionInputs[inputIndex]), DataTypes.Int32)[dimIndex] <= seg;

                // select var value for current segment
                var varInfo = varValues.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value[i]));
                var thenBody = current + 1 < limit
                    ? Split(fusionBody, fusionVars, info, current + 1, limit, varValues, fusionInputs, varMap, fusionInputData, fusionInputsShape)
                    : MakeSplitEntry(fusionBody, fusionVars, varMap, varInfo, fusionInputData, fusionInputsShape, fusionInputs, _relPath, seg);
                var elseBody = sum;
                i++;
                var result = new If(cond, thenBody, elseBody);
                return result;
            });

        // let args
        if (body is If @if)
        {
            return @if.With(paramList: fusionInputs);
        }

        return body;
    }
}

public class FindVar : ExprVisitor<Expr, Unit>
{
    public HashSet<Var> Vars { get; set; } = new();

    // todo: if visit call(VarFusion), then return EffectVar
    protected override Expr VisitLeafVar(Var expr)
    {
        Vars.Add(expr);
        return expr;
    }

    protected override Expr DefaultVisitLeaf(Expr expr) => expr;
}

public abstract class MergeFusionBase : RewriteRule<Pattern>
{
    protected static readonly Dictionary<RuntimeTypeHandle, int> OpList = new()
    {
        { typeof(Reshape).TypeHandle, 0 },
        { typeof(Unsqueeze).TypeHandle, 0 },
        { typeof(Squeeze).TypeHandle, 0 },

        // { typeof(Slice).TypeHandle, 0 },
        { typeof(Concat).TypeHandle, 0 },
        { typeof(Cast).TypeHandle, 0 },
        { typeof(IR.Tensors.Stack).TypeHandle, 0 },
        { typeof(Expand).TypeHandle, 0 },
        { typeof(ConstantOfShape).TypeHandle, 0 },
        { typeof(Where).TypeHandle, 0 },
        { typeof(Compare).TypeHandle, 0 },
        { typeof(Gather).TypeHandle, 0 },

        // compute
        { typeof(Transpose).TypeHandle, 1 },
        { typeof(Unary).TypeHandle, 1 },
        { typeof(Binary).TypeHandle, 2 },
        { typeof(Clamp).TypeHandle, 2 },
        { typeof(Pad).TypeHandle, 2 },

        // ...
        { typeof(Conv2D).TypeHandle, 2 },
        { typeof(MatMul).TypeHandle, 2 },
        { typeof(Tile).TypeHandle, 0 },
    };

    protected int Counter { get; set; }

    public static bool AllConst(Call originCall)
    {
        // 暂时只能合并其他输入为const的
        if (originCall.Arguments.Length == 1)
        {
            return true;
        }

        var otherArgs = originCall.Arguments[1..].ToArray();
        if (otherArgs.All(x => x is Const))
        {
            return true;
        }

        return false;
    }

    public bool ValidTarget(Expr target)
    {
        if (target is ActivationOp)
        {
            return true;
        }

        if (OpList.TryGetValue(target.GetType().TypeHandle, out _))
        {
            return true;
        }

        return false;
    }
}

[RuleGenerator]
public partial class MergeNextMarkerToFusion : MergeFusionBase
{
    // 用于将fusion call 外部的marker合并进来
    public override Pattern Pattern => IsRangeOfMarker("marker", new MergeNextCallToFusion().FusionCall, IsWildcard());

    // 外部保留marker给下一个使用
    public Expr? GetReplace(Marker marker, Call fusionOuterCall, BucketFusion fusion, RunPassContext context)
    {
        if (fusion.Body is Marker)
        {
            return null;
        }

        var result =
            marker.With(target: fusionOuterCall.With(target: fusion.With(body: marker.With(target: fusion.Body))));
        DumpIR(result, $"{Counter++}_{fusion.Name}");
        return result;
    }
}

[RuleGenerator]
public partial class MergePrevMarkerToFusion : MergeFusionBase
{
    public override Pattern Pattern => IsCall(
        "fusionOuterCall",
        IsFusion(
            "fusion",
            "stackvm",
            IsWildcard(),
            GenerateParameters(null)),
        GenerateParameters(null, IsRangeOfMarker("marker", IsWildcard(), IsWildcard())));

    // fusion(marker(xx)) { var } -> fusion(marker(xx)) { marker(var) }
    public Expr? GetReplace(Marker marker, Call fusionOuterCall, BucketFusion fusion)
    {
        var hasVisited = fusion.Parameters[0].Users.Where(u => u is not Fusion).All(u => u is Marker);
        if (hasVisited)
        {
            return null;
        }

        // 不更改原始的input中的marker，要拷贝到fusion里面，将所有的var替换为marker(var)
        // 同时将fusion的body中用到原始var的地方替换为marker(var)
        // MergeCall的时候是支持marker的
        var newBody = ReplaceExpr(fusion.Body, fusion.Parameters[0], marker.With(target: fusion.Parameters[0]));

        // 重新构建fusion
        var newFusion = fusion.With(body: newBody);

        // 返回新的call
        DumpIR(newFusion, $"{Counter++}_{fusion.Name}");
        return fusionOuterCall.With(target: newFusion);
    }
}

[RuleGenerator]
public partial class MergeNextCallToFusion : MergeFusionBase
{
    public Pattern FusionCall => IsCall(
        "fusionOuterCall",
        IsFusion(
            "fusion",
            "stackvm",
            IsWildcard(),
            GenerateParameters(null)),
        GenerateParameters(null));

    public override Pattern Pattern => IsCallWildcard(
        "nextCall",
        IsWildcard("target"),
        IsAlt(
            "maybeFusionCallMarker",
            FusionCall,
            IsRangeOfMarker(FusionCall, IsWildcard())));

    // nextCall(fusion(x)) -> fusion(nextCall(x))
    // nextCall(marker(fusion(x))) -> fusion(nextCall(marker(x)))
    public Expr? GetReplace(Call nextCall, Expr maybeFusionCallMarker, Expr target, Call fusionOuterCall, BucketFusion fusion)
    {
        if (!ValidTarget(target))
        {
            return null;
        }

        // todo: only for single input, effect var must be same
        if (MultiUser(nextCall))
        {
            return null;
        }

        if (!AllConst(nextCall))
        {
            return null;
        }

        DumpIR(nextCall, $"{Counter}_{fusion.Name}_{target.GetType().Name}_origin");

        // 将call里面call fusion的部分替换为fusion的body
        var oldBody = fusion.Body;

        // 附加fusion call外面的marker
        if (maybeFusionCallMarker is Marker fusionMarker)
        {
            oldBody = fusionMarker.With(target: oldBody);
        }

        var newBody = ReplaceExpr(nextCall, fusionOuterCall, oldBody);

        // 除了第一个参数的部分，其他参数可能会用到外面的东西，是不是可以作为var直接传进来??但是这会影响后面ToFusion的部分...

        // 更新fusion的body
        var newFusion = fusion.With(body: newBody);

        // 创建新的call，target为fusion，参数为fusion的参数 // todo:针对非const的情况要处理这里
        // 但是第一个参数要注意，如果有marker那么需要处理marker
        var newArgs = fusionOuterCall.Arguments.ToArray();
        var call = (Expr)nextCall.With(target: newFusion, arguments: newArgs);

        // 附加next call的外面marker
        DumpIR(call, $"{Counter++}_{fusion.Name}_{target.GetType().Name}_after");
        return call;
    }

    private static bool MultiUser(Call nextCall)
    {
        // Marker(LeakyRelu(Marker(Fusion)))
        // 如果user > 1
        if (nextCall.Users.Count > 1)
        {
            return true;
        }

        // 只有一个user也可能是一个marker
        if (nextCall.Users.First() is Marker m)
        {
            // 判断marker的user
            if (m.Users.Count > 1)
            {
                return true;
            }
        }

        // 不是marker那就没问题，一定不是多个user
        return false;
    }

    private bool SameEffectVar(Call originCall, Fusion fusion)
    {
        var array = MakeEffectVarArray(
            CompileSession.CompileOptions.ShapeBucketOptions.VarMap,
            originCall.Arguments[^1..].ToArray());
        if (fusion is BucketFusion varFusion)
        {
            if (array.Length != 0 && !Enumerable.SequenceEqual(varFusion.EffectVar, array))
            {
                return true;
            }
        }
        else
        {
            return true;
        }

        return false;
    }
}

[RuleGenerator]
public partial class MergePrevCallToFusion : MergeFusionBase
{
    public override Pattern Pattern => IsCall(
        "fusionOuterCall",
        IsFusion(
            "fusion",
            "stackvm",
            IsWildcard(),
            GenerateParameters(null)),
        GenerateParameters(
            null,
            MaybeMarker("lhsArg", PrevCall("lhs"))));

    public Pattern PrevCall(string prefix) => IsCallWildcard($"{prefix}PrevCall", IsWildcard($"{prefix}Target"));

    public Pattern MaybeMarker(string exprName, Pattern exprPatten) => IsAlt(
        exprName,
        IsRangeOfMarker(exprPatten, IsWildcard()),
        exprPatten);

    // 输入必须匹配marker，因为即便合并marker也是要在外面保留一份副本
    // fusion(marker(prevCall()) { var } -> fusion(var) { marker(prevCall()) }
    // fusion((prevCall()) { var } -> fusion(var) { prevCall() }

    // dfs
    // xx | marker(xx)不行, 会先匹配到xx
    // xx(marker) | xx 可以
    public Expr? GetReplace(Call fusionOuterCall, BucketFusion fusion, Call lhsPrevCall, Expr lhsTarget, Expr lhsArg)
    {
        if (IsInvalid(lhsPrevCall, lhsTarget))
        {
            return null;
        }

        var (prevCalls, prevOutputMaybeMarker) = CollectInputsInfo(fusionOuterCall, lhsPrevCall, lhsArg);

        DumpIR(fusionOuterCall, $"{Counter}_{lhsTarget.GetType().Name}_{fusion.Name}_origin");

        // todo: 判断rhs的effect var才行
        // 新的fusion var
        var newVars = prevCalls.Select(arg => new Var(arg.Arguments[0].CheckedType)).ToArray();

        // 针对prevCall的所有argument进行替换, prevCall会带入原先var的位置，因此需要继续携带自己的marker。
        // 包含输入1，输入2
        var newPrevCalls = prevCalls.Select((prevCall, i) =>
        {
            // 每个prevCall的arg0替换为fusionVar
            var oldArgs = prevCall.Arguments.ToArray();
            var newArg = oldArgs[0] is Marker marker ? (Expr)marker.With(target: newVars[i]) : newVars[i];
            var newArgs = ReplaceItems(oldArgs, (0, newArg));

            // newArgs，是var，但是可能需要保存var自身的range
            var newPrevCall = prevCall.With(arguments: newArgs);
            return prevOutputMaybeMarker[i] is Marker m ? (Expr)m.With(target: newPrevCall) : newPrevCall;
        }).ToArray();

        // 新的fusion body将原来的var换成prevCall
        var newBody = fusion.Parameters.ToArray().Zip(newPrevCalls).Aggregate(fusion.Body, (sum, pair) =>
        {
            // 此时prevCall携带新的var
            var (fusionVar, newPrevCall) = pair;
            return ReplaceExpr(sum, fusionVar, newPrevCall);
        });

        // 新的fusion的param更换为新的var
        var newParams = ReplaceItems(fusion.Parameters.ToArray(), newVars.Select((v, i) => (i, (Expr)v)).ToArray())
            .OfType<Var>().ToArray();
        var newFusion = fusion.With(body: newBody, parameters: newParams);

        // 新的args为原来所有的prev call的arg[0]
        var newArgs = ReplaceItems(
            fusionOuterCall.Arguments.ToArray(),
            prevCalls.Select((e, i) =>
            {
                return (i, e.Arguments[0]);
            }).ToArray());

        // 原始的fusion的call更换target为新的fusion，以及arg0替换为prevCall的arg0，其他不变
        var call = fusionOuterCall.With(target: newFusion, arguments: newArgs);
        DumpIR(call, $"{Counter++}_{lhsTarget.GetType().Name}_{fusion.Name}_after");
        return call;
    }

    private (Call[] PrevCalls, Expr[] MayBeMarkers) CollectInputsInfo(Call fusionOuterCall)
    {
        var result = fusionOuterCall.Arguments.ToArray().Select(arg =>
        {
            var rhsArg = arg;
            if (rhsArg is Marker marker && marker.Target is Call rhsPrevCall)
            {
                var rhsTarget = rhsPrevCall.Target;

                if (!IsInvalid(rhsPrevCall, rhsTarget))
                {
                    return (rhsPrevCall, (Expr)marker);
                }
            }

            if (rhsArg is Call rhsCall)
            {
                var rhsTarget = rhsCall.Target;

                if (!IsInvalid(rhsCall, rhsTarget))
                {
                    return (rhsCall, rhsCall);
                }
            }

            throw new NotImplementedException();
        }).ToArray();
        var prevCalls = result.Select(x => x.rhsPrevCall).ToArray();
        var maybeMarkers = result.Select(x => x.Item2).ToArray();
        return (prevCalls, maybeMarkers);
    }

    private (Call[] PrevCalls, Expr[] MayBeMarkers) CollectInputsInfo(Call fusionOuterCall, Call lhsPrevCall, Expr lhsArg)
    {
        var prevCalls = new[] { lhsPrevCall };
        var argsMarker = new[] { lhsArg };
        if (fusionOuterCall.Arguments.Length > 1)
        {
            var rhsArg = fusionOuterCall.Arguments[1];
            if (rhsArg is Marker marker && marker.Target is Call rhsPrevCall)
            {
                var rhsTarget = rhsPrevCall.Target;

                if (!IsInvalid(rhsPrevCall, rhsTarget))
                {
                    prevCalls = prevCalls.Append(rhsPrevCall).ToArray();
                    argsMarker = argsMarker.Append(rhsArg).ToArray();
                }
            }

            if (rhsArg is Call rhsCall)
            {
                var rhsTarget = rhsCall.Target;

                if (!IsInvalid(rhsCall, rhsTarget))
                {
                    prevCalls = prevCalls.Append(rhsCall).ToArray();
                    argsMarker = argsMarker.Append(rhsCall).ToArray();
                }
            }
        }

        return (prevCalls, argsMarker);
    }

    private bool IsInvalid(Call lhsPrevCall, Expr lhsTarget)
    {
        if (lhsPrevCall.Users.Count > 1)
        {
            return true;
        }

        if (!ValidTarget(lhsTarget))
        {
            return true;
        }

        if (!AllConst(lhsPrevCall))
        {
            return true;
        }

        return false;
    }
}

internal record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);

internal static class ShapeBucketHelper
{
    public static Marker[] GetCallInputs(Call call) =>
        call.Arguments.ToArray().OfType<Marker>().Where(x => x.Target is not TensorConst).ToArray();

    public static Var[] MakeEffectVarArray(Dictionary<Var, Expr[]> varMap, params Expr[] args)
    {
        var visitor = new FindVar();
        args.ForEach(arg =>
        {
            var argShapeExpr = arg.EvaluateShapeExpr(varMap);
            visitor.Visit(argShapeExpr);
        });
        return visitor.Vars.ToHashSet().Except(varMap.Keys.ToHashSet()).ToHashSet().ToArray();
    }

    internal static void DumpIR(Expr expr, string prefix, string? reletivePath = null)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
        {
            DumpScope.Current.DumpIR(expr, prefix, reletivePath);
        }
    }
}
