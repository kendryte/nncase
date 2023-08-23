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
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeExpr;
using Nncase.Passes.Transforms;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
using static Nncase.Passes.Rules.ShapeBucket.ShapeBucketHelper;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;
using Dimension = Nncase.IR.Dimension;
using FoldConstCall = Nncase.Passes.Mutators.FoldConstCall;
using Stack = Nncase.IR.Tensors.Stack;
using Tuple = System.Tuple;

namespace Nncase.Passes.Rules.ShapeBucket;

public class BucketFusion : Fusion, IEquatable<BucketFusion>
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
            // todo: change list
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

    public bool Equals(BucketFusion? other)
    {
        if (other == null)
        {
            return false;
        }

        return Name == other.Name && ModuleKind == other.ModuleKind && Body.GetHashCode() == other.Body.GetHashCode() &&
            Parameters.SequenceEqual(other.Parameters) && EffectVar.SequenceEqual(other.EffectVar);
    }

    public override bool Equals(object? obj)
    {
        return Equals(obj as BucketFusion);
    }
}

[RuleGenerator]
public partial class CallToFusion : RewriteRule<Pattern>
{
    private readonly bool _onlyDynamic;

    public CallToFusion(bool onlyDynamic)
    {
        _onlyDynamic = onlyDynamic;
    }

    public CallToFusion()
    {
        _onlyDynamic = false;
    }

    public static int Counter { get; set; }

    public string ModuleKind => "stackvm";

    public override Pattern Pattern => throw new InvalidOperationException();

    protected virtual bool MustHaveMarker => true;

    private Call? CurrentCall { get; set; }

    private string Name => CurrentCall!.Target.GetType().Name;

    private string RelPath => $"{Counter}_{CurrentCall!.Target.GetType().Name}";

    public virtual bool Check(Call call)
    {
        return true;
    }

    public Expr? GetReplace(Call call, IMatchResult matchResult)
    {
        // 第二轮的时候再开
        if (_onlyDynamic && call.CheckedShape.IsFixed)
        {
            return null;
        }

        var originType = call.CheckedType;
        CurrentCall = call;
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
        Counter++;

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

    protected virtual Expr ProcessForNewBody(Var[] fusionVars, Expr[] args, Expr expr) => expr;

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

    protected virtual void Init(IMatchResult result)
    {
    }

    protected virtual Expr ReplaceVarsWithArg(Var[] fusionVars, Expr[] args, Expr newCall) =>
        fusionVars.Zip(args).Aggregate(newCall, (newBody, tuple) =>
        {
            var (fusionVar, arg) = tuple;
            return ReplaceUtility.ReplaceExpr(newBody, arg, fusionVar);
        });

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
        var body = ReplaceVarsWithArg(fusionVars, args, newCall);

        var f = new BucketFusion($"{Name}_{Counter}", ModuleKind, set, body, fusionVars);
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
                    return (originIndex, arg: m.With(target: pair.First));
                }

                return (originIndex, arg: (Expr)pair.First);
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
    public MarkerCallToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }

    public MarkerCallToFusion()
        : base(false)
    {
    }

    public override Pattern Pattern => IsRangeOfMarker(
        "callMarker",
        IsCallWildcard("call", IsOp<T>()),
        IsTensorConst());

    protected Marker? CallMarker { get; set; }

    protected override Expr ProcessForNewBody(Var[] fusionVars, Expr[] args, Expr expr) =>
        CallMarker!.With(target: expr);

    protected override Expr ProcessForOuterCall(Expr expr) => CallMarker!.With(target: expr);

    protected override void Init(IMatchResult result)
    {
        CallMarker = (Marker)result["callMarker"];
    }
}

public class MultiUserCallToFusion : CallToFusion
{
    public MultiUserCallToFusion(bool onlyDynamic = false)
        : base(onlyDynamic)
    {
    }

    public MultiUserCallToFusion()
    {
    }

    public override Pattern Pattern => IsWildcard("call", expr =>
    {
        if (expr is Call c && c.Target is not BucketFusion)
        {
            if (c.Target is Binary)
            {
                if (c.Arguments[0] is not Const && c.Arguments[1] is not Const)
                {
                    return false;
                }

                return true;
            }

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

    public override bool Check(Call call)
    {
        return !call.Users.ToArray().OfType<Var>().Any();
    }

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
}

public class Conv2DToFusion : MarkerCallToFusion<Conv2D>
{
    public Conv2DToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }

    public Conv2DToFusion()
    {
    }
}

// tflite相比于onnx的比较特殊，output shape是原图进行计算的，而不是自行创建表达式计算。
// 如果采用一样的处理方法会导致复制输入中的function和call
// 对于tflite的所有反卷积的通用性不能确保，暂且这样硬编码，另外tflite的动态shape也很少见
// 这里本质的问题是因为output shape所指向的很可能并不是input，或者说是input并不是output shape所指向的表达式的子表达式
public class TFConv2DTransposeToFusion : MarkerCallToFusion<Conv2DTranspose>
{
    private Call? _transpose;

    private Call? _originCall;

    private Marker? _transposeInputMarker;

    public TFConv2DTransposeToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }

    public override Pattern Pattern => IsRangeOfMarker(
        "callMarker",
        IsCallWildcard(
            "call",
            IsOp<Conv2DTranspose>(),
            IsRangeOfMarker(
                IsCallWildcard(
                    "transpose",
                    IsOp<Transpose>(),
                    IsRangeOfMarker(
                        "transposeInputMarker",
                        IsCallWildcard("originCall", IsWildcard(), IsWildcard()),
                        IsWildcard())),
                IsWildcard())),
        IsTensorConst());

    protected override (Expr, int)[] CollectInputs(Call call)
    {
        return new[] { (_originCall!.Arguments[0], 0) };
    }

    protected override void Init(IMatchResult result)
    {
        _transpose = (Call)result["transpose"];
        _originCall = (Call)result["originCall"];
        _transposeInputMarker = (Marker)result["transposeInputMarker"];
        base.Init(result);
    }

    protected override Expr ReplaceVarsWithArg(Var[] fusionVars, Expr[] args, Expr newCall)
    {
        var convTranspose = (Call)CallMarker!.Target;
        var c = ReplaceCallFirstParam(
            convTranspose,
            _transposeInputMarker!.With(target:
                ReplaceCallFirstParam(
                    _transpose!,
                    _transposeInputMarker.With(target:
                        ReplaceCallFirstParam(_originCall!, fusionVars[0])))));
        return CallMarker.With(target: base.ReplaceVarsWithArg(fusionVars, args, c));
    }

    protected override Expr ProcessForNewBody(Var[] fusionVars, Expr[] args, Expr expr)
    {
        // 1. reconstruct new body

        // 2. replace
        var newBody = fusionVars.Zip(args).Aggregate(expr, (newBody, tuple) =>
        {
            var (fusionVar, arg) = tuple;
            return ReplaceUtility.ReplaceExpr(newBody, arg, fusionVar);
        });
        return CallMarker!.With(target: newBody);

        // return ReplaceClone(callMarker.With(target: newBody), fusionVars.Zip(args).ToArray());
    }
}

public class Conv2DTransposeToFusion : MarkerCallToFusion<Conv2DTranspose>
{
    public Conv2DTransposeToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }

    // when OutputShape is Const, it means output shape is not effected by input.
    public override bool Check(Call call) => call.Arguments[Conv2DTranspose.OutputShape.Index] is not Const;
}

public class MatmulToFusion : MarkerCallToFusion<MatMul>
{
    public MatmulToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }
}

public class ActToFusion : MarkerCallToFusion<ActivationOp>
{
    public ActToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }
}

public class TransposeToFusion : MarkerCallToFusion<Transpose>
{
    public TransposeToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }

    protected override bool MustHaveMarker => false;
}

public class UnaryToFusion : MarkerCallToFusion<Unary>
{
    public UnaryToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }

    public UnaryToFusion()
    {
    }
}

// todo: do more check for binary
public class BinaryToFusion : MarkerCallToFusion<Binary>
{
    public BinaryToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }

    // public override bool Check(Call call) => call.CheckedShape.Rank > 1;
}

[RuleGenerator]
public partial class ClearRequire : RewriteRule<Pattern>
{
    // for require(true, value, msg)
    public override Pattern Pattern { get; } =
        IsRequire(require => true, IsTensorConst("predicate"), IsWildcard("expr"));

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

public class FusionBucketContext
{
    public FusionBucketContext(Call outerCall, BucketFusion fusion, Dictionary<Var, Expr[]> varMap, Dictionary<Var, int[]> dimVarValues, ShapeExprCache cache)
    {
        OuterCall = outerCall;
        Fusion = fusion;
        VarMap = varMap;
        Cache = cache;
        Cache.VarMap = varMap;
        FusionInputShapeExpr = MakeFusionInputShapeExpr(outerCall, fusion, cache);
        CheckAlive(FusionInputShapeExpr);
        DimVarValues = dimVarValues;
        Arguments = OuterCall.Arguments.ToArray();
        Parameters = Fusion.Parameters.ToArray();
        FixedShapeCache = new();
        SliceShape = ComputeSliceShape();
    }

    public Expr SliceShape { get; }

    public Call OuterCall { get; }

    public BucketFusion Fusion { get; }

    public Dictionary<Var, Expr[]> VarMap { get; }

    public Dictionary<Var, Expr[]> FusionInputShapeExpr { get; }

    public Dictionary<Var, int[]> DimVarValues { get; }

    public Expr[] Arguments { get; }

    public Var[] Parameters { get; }

    public ShapeExprCache Cache { get; }

    // segIndex -> fixed shape list
    public Dictionary<int, int[][]> FixedShapeCache { get; }

    public Expr FusionBody => Fusion.Body;

    public Dictionary<Var, IValue> DimVarValue(int i) =>
        DimVarValues.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value[i]));

    // ShapeOf而不是shape表达式，用于计算Slice的shape
    private static Dictionary<Var, Expr[]> MakeShapeOfFusionInput(Var[] parameters, Expr[] args)
    {
        var fusionInputShapes = parameters
            .Zip(args)
            .ToDictionary(pair => pair.First, pair =>
            {
                var shape = Cast((Expr)ShapeOf(pair.Second), DataTypes.Int32);
                return Enumerable.Range(0, pair.Second.CheckedShape.Rank).Select(i => shape[i]).ToArray();
            });
        return fusionInputShapes;
    }

    private static Dictionary<Var, Expr[]> MakeFusionInputShapeExpr(Call call, BucketFusion fusion, ShapeExprCache cache)
    {
        var data = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray().Select((arg, i) =>
        {
            // DumpIR(arg, "MakeFusionInputShapeExprArg");
            var result = arg.EvaluateShapeExpr(cache);
            if (!result.InferenceType())
            {
                DumpIR(result, "InvalidInputShapeExpr");
                throw new InvalidOperationException();
            }

            return Enumerable.Range(0, arg.CheckedShape.Rank).Select(i =>
            {
                var res = result[i];
                return res;
            }).ToArray();
        })).Select(pair => new KeyValuePair<Var, Expr[]>(pair.First, pair.Second));
        var fusionInputData = data.ToDictionary(pair => pair.Key, pair => pair.Value);
        return fusionInputData;
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

    private Expr ComputeSliceShape()
    {
        var originBody = FusionBody;
        _ = MakeShapeOfFusionInput(Parameters, Arguments);
        var originShape = originBody.EvaluateShapeExpr(FusionInputShapeExpr);
        originShape.InferenceType();
        return originShape;
    }
}

[RuleGenerator]
public partial class FusionBucket : RewriteRule<Pattern>
{
    private static int _counter;

    private static string _relPath = string.Empty;

    private readonly ShapeExprCache _cache = ShapeExprCache.Default;

    public FusionBucket(Dictionary<BucketFusion, FusionShapeData[]> list)
    {
        FusionShapeInfo = list;
    }

    public Dictionary<BucketFusion, FusionShapeData[]> FusionShapeInfo { get; set; }

    public override Pattern Pattern => IsCall(
        "outerCall",
        IsFusion(
            "fusion",
            "stackvm",
            IsWildcard("fusionBody"),
            GenerateParameters(null)),
        GenerateParameters(null));

    internal Dictionary<Var, Expr[]> VarMap => CompileSession.CompileOptions.ShapeBucketOptions.VarMap;

    public static Expr PreProcess(FusionBucketContext context, Var param, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, IValue> varValues, Dictionary<Var, Expr[]> fusionInputData, int segIndex, int inputIndex)
    {
        // Console.WriteLine($"seg index{segIndex}");
        if (context.FixedShapeCache.TryGetValue(segIndex, out var cachedFixedShape))
        {
            // var cachedShape = cachedFixedShape[inputIndex];
            // Console.WriteLine(string.Join(",", cachedShape));
            // Console.WriteLine("Cache ok");
            return new Call(new BucketPad(), param, cachedFixedShape[inputIndex]);
        }

        throw new InvalidDataException("Shape Cache not found");
    }

    public static (Dictionary<Var, IValue> MinDict, Dictionary<Var, IValue> MaxDict) GetBoundDict(
        Dictionary<Var, Expr[]> inputInfo, Dictionary<string, (int Min, int Max)> rangeInfo)
    {
        // find vars in Input ShapeExpr
        var vars = inputInfo.Values.SelectMany(x => x).OfType<Var>().ToHashSet().ToArray();

        // DimVarName -> Dict.key -> Dict.Value
        var minDict = rangeInfo.ToDictionary(
            pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => (IValue)Value.FromTensor(pair.Value.Min));
        var maxDict = rangeInfo.ToDictionary(
            pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => (IValue)Value.FromTensor(pair.Value.Max));
        return (minDict, maxDict);
    }

    public static Expr MakeSplitEntry(FusionBucketContext context, Dictionary<Var, IValue> varInfo, int segIndex)
    {
        var originBody = context.FusionBody;
        var fusionVars = context.Parameters;
        var fixInputs = fusionVars
            .Select((arg, i) =>
                PreProcess(context, arg, context.VarMap, varInfo, context.FusionInputShapeExpr, segIndex, i)).ToArray();

        // 替换逻辑：新的body中的var -> fusion原始的var -> target为fusion的call的input
        // 本质上只是对这个body的所有输入做替换
        // 避免这里的修改影响到原始的body，每个分支需要进行自己的修改，所以要clone处理
        // DumpIR(originBody, "originBody", _relPath);
        var call = ReplaceClone(originBody, fusionVars.Zip(fixInputs).ToArray());
        if (!call.InferenceType())
        {
            DumpIR(call, "InvalidType");
            throw new InvalidOperationException();
        }

        var slice = MakeSlice(context, call, originBody);
        DumpIR(slice, $"slice_{segIndex}", _relPath);
        return slice;
    }

    public Expr? GetReplace(Call outerCall, BucketFusion fusion, Expr fusionBody)
    {
        if (ShouldRestore(outerCall, fusion))
        {
            return RestoreBodyWithArgs(outerCall.Arguments.ToArray(), fusion.Parameters.ToArray(), fusion.Body);
        }

        fusionBody = CompilerServices.Rewrite(
            fusionBody,
            new IRewriteRule[]
            {
                new FoldStackGetItem(), new FoldShapeOf(), new FoldTwoReshapes(), new FoldTwoCasts(),
                new FoldTwoSlices(), new FoldNopBinary(), new FoldNopCast(), new Neutral.FoldConstCall(),
                new FoldNopReshape(), new FoldNopSlice(), new FoldIf(),
            },
            new());
        Console.WriteLine($"FusionBucketGetReplace {_counter} {fusion.Name}");
        _relPath = $"{_counter}";

        DumpIR(outerCall, $"BucketOriginFusion_{fusion.Name}", _relPath);

        var options = CompileSession.CompileOptions.ShapeBucketOptions;
        var dimVarValues = MakeVarValuesForAllSegment(options);
        var context = new FusionBucketContext(outerCall, fusion, VarMap, dimVarValues, _cache);

        var shapeInfos = Array.Empty<FusionShapeData>();
        if (!FusionShapeInfo.TryGetValue(fusion, out shapeInfos))
        {
            // todo: 不知道为什么有的时候无法从key中获取
            var list = FusionShapeInfo.Where(x => x.Key == fusion).ToArray();
            if (list.Length != 1)
            {
                throw new InvalidOperationException($"NoKey{fusion.Name}");
            }

            shapeInfos = list[0].Value;
        }

        var allFixedShapes = shapeInfos
            .Select(x =>
                x.InputShapes.Select(iShape => iShape.AsTensor().ToArray<int>().ToArray()).ToArray()).ToArray();
        for (int i = 0; i < shapeInfos.Length; i++)
        {
            for (int j = 0; j < allFixedShapes.Length; j++)
            {
                context.FixedShapeCache[j] = allFixedShapes[j];
            }
        }

        // reverse
        var minFixedShapeList = allFixedShapes[^1];
        var maxFixedShapeList = allFixedShapes[0];

        PrintMinMaxShape(minFixedShapeList, maxFixedShapeList, _relPath);

        // 2. get dim info(inputIndex, (dimIndex, range)
        var counts = ComputeCounts(minFixedShapeList, maxFixedShapeList, out int totalCount);
        if (IsFixed(totalCount, minFixedShapeList, maxFixedShapeList))
        {
            var fix = FixInput(context, minFixedShapeList);
            DumpIR(fix, "BucketResultFix", _relPath);
            _counter++;
            return fix;
        }

        // todo: process total count, matmul maybe multi count, but other should check this
        if (totalCount > 1)
        {
            // Console.WriteLine($"{fusion.Name} totalCount > 1");
        }

        var info = ComputeSegmentInfo(counts, options);
        var body = Split(context, info);
        body.InferenceType();

        if (body.Users.Count > 1 || body.CheckedType is InvalidType)
        {
            throw new InvalidOperationException();
        }

        if (body is not If)
        {
            _counter++;
            DumpIR(body, "Rebuild", _relPath);
            return body;
        }

        // DumpIR(body, "newBodyBeforeReplace", _relPath);
        // FixInput Replace Var
        var newBody = ReplaceFusionVarWithCallArgs(fusion, context.Arguments, body);

        // let bind
        if (newBody is If @if)
        {
            newBody = IR.F.Math.Require(true, @if.With(paramList: context.Arguments));
        }

        DumpIR(newBody, "BucketResult", _relPath);
        _counter++;
        if (newBody.CheckedType is InvalidType)
        {
            throw new InvalidOperationException("InvalidBucketBody");
        }

        return newBody;
    }

    private static void PrintShapeInfos(FusionShapeData[] shapeInfos)
    {
        for (var i = 0; i < shapeInfos.Length; i++)
        {
            Console.WriteLine($"Segment Index {i}");
            var inShapes = shapeInfos[i].InputShapes;
            for (int j = 0; j < inShapes.Length; j++)
            {
                var shape = inShapes[j].AsTensor().ToArray<int>();
                Console.WriteLine($"Input {j} shape:");
                Console.WriteLine(string.Join(",", shape));
            }
        }
    }

    public Expr FixInput(FusionBucketContext context, int[][] shapeList)
    {
        var fixedShapeInput = context.Arguments.Zip(shapeList).Select(pair =>
        {
            var (arg, fixedShape) = pair;
            return (Expr)new Call(new FixShape(), arg, fixedShape);
        }).ToArray();
        return ReplaceClone(context.FusionBody, context.Parameters.Zip(fixedShapeInput).ToArray());
    }

    private static Expr MakeSlice(FusionBucketContext context, Expr call, Expr originBody)
    {
        if (call.CheckedType is TupleType tuple)
        {
            var fields = Enumerable.Range(0, tuple.Count)
                .Select(i => MakeSliceForTensor(originBody[i], call[i], context)).ToArray();
            return new IR.Tuple(fields);
        }

        return MakeSliceForTensor(originBody, call, context);
    }

    private static Expr MakeSliceForTensor(Expr originBody, Expr call, FusionBucketContext context)
    {
        var sliceShape = context.SliceShape;
        var rank = call.CheckedShape.Rank;
        var body = (Expr)Slice(call, Enumerable.Repeat(0, rank).ToArray(), Cast(sliceShape, DataTypes.Int32), rank);
        return body;
    }

    private static bool IsFixed(int totalCount, int[][] minFixedShapeList, int[][] maxFixedShapeList) =>
        totalCount == 0 || (minFixedShapeList[0].SequenceEqual(maxFixedShapeList[0]) &&
                            minFixedShapeList[1].SequenceEqual(maxFixedShapeList[1]));

    private static bool ShouldRestore(Call outerCall, BucketFusion fusion)
    {
        return fusion.IsSimple ||
               outerCall.CheckedType is TupleType ||
               outerCall.CheckedShape.Rank == 0 ||
               outerCall.Arguments.ToArray().Any(arg =>
                   arg.CheckedType is TupleType);
    }

    private static Expr RestoreBodyWithArgs(Expr[] args, Var[] parameters, Expr body) =>
        ReplaceClone(body, parameters.Zip(args).ToArray());

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

    private static (int InputIndex, (int First, (int First, int Second) Second)[] Range)[] ComputeCounts(
        int[][] minFixedShapeList, int[][] maxFixedShapeList, out int totalCount)
    {
        (int InputIndex, (int First, (int First, int Second) Second)[] Range)[] counts = minFixedShapeList
            .Zip(maxFixedShapeList).Select((pair, inputIndex) =>
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

    private static Expr Split(FusionBucketContext context, SegmentInfo info)
    {
        var fusionInputs = context.Arguments;
        var (inputIndex, dimIndex, segments) = info;
        var dim = ShapeOf(fusionInputs[inputIndex])[dimIndex];
        var failure = MakeFailure(context.FusionBody);

        int i = 0;

        // 1. 普通情况不应该rebuild
        // 2. rebuild的正确性
        // if (ShouldBeRebuild(context))
        // {
        //     Console.WriteLine("Rebuild");
        //     return RestoreBodyWithArgs(context.Arguments, context.Parameters, context.FusionBody);
        // }
        var body = segments.OrderByDescending(x => x).Aggregate(
            failure,
            (sum, seg) =>
            {
                // 根据var，也就是target为这个fusion的call的参数来进行判断落在哪个段
                var cond = dim <= (long)seg;

                // select var value for current segment
                var varInfo = context.DimVarValue(i);
                var thenBody = MakeSplitEntry(context, varInfo, i);
                var elseBody = sum;
                i++;

                var result = new If(cond, thenBody, elseBody);
                return result;
            });

        return body;
    }

    private static bool ShouldBeRebuild(FusionBucketContext context)
    {
        var varInfo = context.DimVarValue(0);
        var entry = MakeSplitEntry(context, varInfo, 0);
        return entry switch
        {
            IR.Tuple tuple => tuple.Fields.ToArray().Any(ShouldBeRebuild),
            Call => ShouldBeRebuild(entry),
            _ => throw new ArgumentOutOfRangeException("context"),
        };
    }

    private static bool ShouldBeRebuild(Expr entry) => entry is Call { Target: IR.Tensors.Slice } c &&
                                                       (!c.Arguments[IR.Tensors.Slice.Input.Index].CheckedShape
                                                           .IsFixed);

    private static Expr MakeFailure(Expr fusionBody)
    {
        var failure = fusionBody.CheckedType switch
        {
            TupleType tuple => new IR.Tuple(tuple.Fields.ToArray()
                .Select(x =>
                {
                    return ConstantOfShape(new[] { 1 }, Cast(0, ((TensorType)x).DType));
                }).ToArray()),
            TensorType tensorType => (Expr)ConstantOfShape(new[] { 1 }, Cast(0, tensorType.DType)),
            _ => throw new ArgumentOutOfRangeException("fusionBody"),
        };
        return IR.F.Math.Require(false, failure, "input dim large than limit");
    }
}

internal record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);
