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
using GiGraph.Dot.Types.Geometry;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Toolkit.HighPerformance;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.ShapeExpr;
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
using BaseFunction = Nncase.IR.BaseFunction;
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
    private readonly bool _greedy = true;

    public MultiUserCallToFusion(bool onlyDynamic = false, bool greedy = true)
        : base(onlyDynamic)
    {
        _greedy = greedy;
    }

    public MultiUserCallToFusion()
    {
    }

    public override Pattern Pattern => IsWildcard("call", expr =>
    {
        if (expr is Call c && c.Target is not BucketFusion)
        {
            return CallValidator.ValidTarget(c, _greedy);
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
            convTranspose.Target,
            convTranspose.Arguments.ToArray(),
            _transposeInputMarker!.With(target:
                ReplaceCallFirstParam(
                    _transpose!.Target,
                    _transpose!.Arguments.ToArray(),
                    _transposeInputMarker.With(target:
                        ReplaceCallFirstParam(_originCall!.Target, _originCall!.Arguments.ToArray(), fusionVars[0])))));
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
}

public class ReshapeToFusion : CallToFusion
{
    public ReshapeToFusion(bool isDynamic = false)
        : base(isDynamic)
    {
    }

    public override Pattern Pattern => IsCallWildcard("call", IsOp<Reshape>());

    protected override (Expr, int)[] CollectInputs(Call call)
    {
        var input = call.Arguments[IR.Tensors.Reshape.Input.Index];
        var inputPair = (input, IR.Tensors.Reshape.Input.Index);
        var padPair = (call.Arguments[IR.Tensors.Reshape.Shape.Index], IR.Tensors.Reshape.Shape.Index);
        if (padPair.Item1 is TensorConst)
        {
            return new[] { inputPair };
        }

        return new[] { inputPair, padPair };
    }
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

    public override bool Check(Call call) => call.CheckedShape.Rank > 1;
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
    private readonly int _index;

    public FusionBucketContext(Call outerCall, BucketFusion fusion, ShapeBucketOptions options, ShapeExprCache cache, int index, FusionShapeData[] shapeInfos)
    {
        OuterCall = outerCall;
        Fusion = fusion;
        VarMap = options.VarMap;
        Cache = cache;
        Cache.VarMap = options.VarMap;
        FusionInputShapeExpr = new();
        DimVarValues = MakeVarValuesForAllSegment(options);

        Arguments = OuterCall.Arguments.ToArray();
        Parameters = Fusion.Parameters.ToArray();
        FixedShapeCache = new();
        SliceShape = ComputeSliceShape(shapeInfos);
        _index = index;
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

    private static Expr ReplaceShapeOf(Dictionary<Var, Expr[]> fusionInputsShapeExpr, Dictionary<Var, Expr[]> varMap, Expr originShape, Var[] parameters, Var[] dimVarKeys, FusionBucketContext context, Dictionary<Expr, Var> dict)
    {
        // return originShape;
        // 拷贝shape表达式，以免被原始的计算引用
        var cloneShape = originShape.Clone();
        CompilerServices.Rewrite(cloneShape, new[] { new RemoveMarker() }, new());
        var f = new FindVar();
        f.Visit(cloneShape);
        var newVars = f.Vars;

        // 可能在VarMap里面有，但是newVar中没有，所以把newVar转换为oldVar
        var newDict = fusionInputsShapeExpr
            .Concat(varMap)
            .Where(pair => newVars.FindFirst(newVar => newVar.Name == pair.Key.Name) != null)
            .ToDictionary(
                pair =>
                {
                    var k = newVars.FindFirst(newVar => newVar.Name == pair.Key.Name);
                    return k;
                },
                pair =>
                {
                    var v = pair.Value;
                    return v;
                })
            .ToDictionary(x => x.Key, x => x.Value);

        var originVars = parameters
            .ToArray()
            .Concat(varMap.Keys)
            .Concat(dimVarKeys)
            .ToDictionary(v => v.Name, v => v);

        Task.Run(() => new FoldNopTuple().RunAsync(new Function(cloneShape), new())).Wait();
        Expr sliceShape = cloneShape;
        var p = new ReplaceOfCollector();
        p.Visit(cloneShape);
        var processList = p.List;
        processList.Reverse();

        var argCache = context.Arguments.ToDictionary(arg => arg, arg => (Expr)ShapeOf(arg));
        var exprs = argCache.SelectMany(pair => new[] { pair.Key, pair.Value }).ToArray();
        var pinner = new ExprPinner(exprs);
        var cache = new ShapeExprCache(newDict, argCache);

        foreach (var call in processList)
        {
            var newShapeOf = call.Arguments[0].EvaluateShapeExpr(cache);
            ReplaceUtility.ReplaceAllUsesWith(call, newShapeOf);
        }

        foreach (var (key, value) in dict)
        {
            var mutator = new Passes.Mutators.Substitutor(e =>
            {
                if (e is Var v1 && v1.Name == value.Name)
                {
                    return key;
                }

                return null;
            });
            mutator.Visit(sliceShape, Unit.Default);
        }

        newVars.ToArray().ForEach(newVar =>
        {
            if (originVars.TryGetValue(newVar.Name, out var originVar))
            {
                ReplaceExpr(sliceShape, newVar, originVar);
            }
        });

        var body = sliceShape;
        var simplifySliceShape = SimplifyShape(body);
        return simplifySliceShape;
    }

    private static Expr SimplifyShape(Expr body) =>
        CompilerServices.Rewrite(
            body,
            new IRewriteRule[]
            {
                new FoldStackGetItem(), new FoldShapeOf(), new FoldTwoReshapes(), new FoldTwoCasts(),
                new FoldTwoSlices(), new FoldNopBinary(), new FoldNopCast(), new Neutral.FoldConstCall(),
                new FoldNopReshape(), new FoldNopSlice(), new FoldIf(), new FoldBroadcastShape(), new FoldSplitShapeOf(),
            },
            new());

    private Expr ComputeSliceShape(FusionShapeData[] shapeInfos)
    {
        var originBody = FusionBody;
        var shapeOfFusionInput = MakeShapeOfFusionInput(Parameters, Arguments);
        var originShape = originBody.EvaluateShapeExpr(shapeOfFusionInput);
        originShape.InferenceType();

        // complex check
        // 判断是否需要replace,里面是否存在满足条件的shapeof
        var args = Arguments.ToDictionary(x => x, x => new Var(x.CheckedType));
        var input = MakeShapeOfFusionInput(Parameters, args.Values.ToArray());
        var varShape = originBody.EvaluateShapeExpr(input);
        var p = new ReplaceOfCollector();
        p.Visit(originBody);
        if (p.List.Count == 0)
        {
            return SimplifyShape(originShape);
        }

        return ReplaceShapeOf(shapeOfFusionInput, VarMap, varShape, Parameters, DimVarValues.Keys.ToArray(), this, args);
    }
}

public class ReplaceOfCollector : ExprVisitor<Expr, Unit>
{
    public List<Call> List { get; } = new();

    protected override Expr VisitLeafCall(Call expr)
    {
        var input = expr.Arguments[0];

        // input is marker or call
        if (expr.Target is ShapeOf && input.CheckedShape.Rank > 2 && input is not Var)
        {
            List.Add(expr);
        }

        return expr;
    }

    protected override Expr DefaultVisitLeaf(Expr expr) => expr;
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

    public static Expr PreProcess(FusionBucketContext context, Var param, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, IValue> varValues, Dictionary<Var, Expr[]> fusionInputData, int segIndex, int inputIndex)
    {
        // Console.WriteLine($"seg index{segIndex}");
        if (context.FixedShapeCache.TryGetValue(segIndex, out var cachedFixedShape))
        {
            var shape = cachedFixedShape[inputIndex];
            if ((param.CheckedShape.IsFixed && shape.SequenceEqual(param.CheckedShape.ToValueArray())) || param.CheckedShape.IsScalar)
            {
                return param;
            }

            return new Call(new BucketPad(), param, shape);
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

    public static Expr Split(FusionBucketContext context)
    {
        var failure = MakeFailure(context.FusionBody);

        // todo: test this
        var value = GetVarValue(context);

        int i = 0;

        // todo: only used for same range
        var body = context.DimVarValues.First().Value.OrderByDescending(x => x).Aggregate(
            failure,
            (sum, seg) =>
            {
                // 根据var，也就是target为这个fusion的call的参数来进行判断落在哪个段
                var cond = value <= (long)seg;
                var sameCond = IR.F.Math.Equal(value, (long)seg);

                // select var value for current segment
                var varInfo = context.DimVarValue(i);
                var thenBody = MakeSplitEntry(context, varInfo, i, sameCond);
                var elseBody = sum;
                i++;

                var result = new If(cond, thenBody, elseBody);
                return result;
            });

        return body;
    }

    public static Expr MakeSplitEntry(FusionBucketContext context, Dictionary<Var, IValue> varInfo, int segIndex, Expr sameCond, bool sameOpt = false)
    {
        var originBody = context.FusionBody;
        var fusionVars = context.Parameters;
        var fixInputs = fusionVars
            .Select((arg, i) =>
                PreProcess(context, arg, context.VarMap, varInfo, context.FusionInputShapeExpr, segIndex, i)).ToArray();

        // 替换逻辑：新的body中的var -> fusion原始的var -> target为fusion的call的input
        // 本质上只是对这个body的所有输入做替换
        // 避免这里的修改影响到原始的body，每个分支需要进行自己的修改，所以要clone处理
        var call = ReplaceClone(originBody, fusionVars.Zip(fixInputs).ToArray());
        if (!call.InferenceType())
        {
            DumpIR(call, "InvalidType");
            throw new InvalidOperationException();
        }

        var slice = MakeSlice(context, call, originBody);
        return sameOpt
            ? new If(sameCond, call, slice)
            : slice;
    }

    public static Expr GetVarValue(FusionBucketContext context)
    {
        var varList = context.VarMap.Where(pair => pair.Value.Any(x => x is Var)).Select(pair =>
        {
            var (v, dims) = pair;
            var i = dims.IndexOf(x => x is Var);
            return ShapeOf(v)[i];
        }).ToArray();

        if (varList.Length > 1)
        {
            return varList.Aggregate((sum, x) => IR.F.Math.Max(sum, x));
        }

        return varList.First();
    }

    public Expr? GetReplace(Call outerCall, BucketFusion fusion, Expr fusionBody)
    {
        if (ShouldRestore(outerCall, fusion))
        {
            return RestoreBodyWithArgs(outerCall.Arguments.ToArray(), fusion.Parameters.ToArray(), fusion.Body);
        }

        _relPath = $"{_counter}";

        DumpIR(outerCall, $"BucketOriginFusion_{_counter}_{fusion.Name}", _relPath);

        var options = CompileSession.CompileOptions.ShapeBucketOptions;
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

        // 每个段的output
        var context = new FusionBucketContext(outerCall, fusion, options, _cache, _counter, shapeInfos);

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

        // todo: fix min max
        // reverse
        var minFixedShapeList = allFixedShapes[^1];
        var maxFixedShapeList = allFixedShapes[0];

        // PrintMinMaxShape(minFixedShapeList, maxFixedShapeList, _relPath);

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

        // 1. 普通情况不应该rebuild
        // 2. rebuild的正确性
        if (ShouldBeRebuild(context))
        {
            _counter++;
            Console.WriteLine("Rebuild");
            var rebuild = RestoreBodyWithArgs(context.Arguments, context.Parameters, context.FusionBody);
            DumpIR(rebuild, "Rebuild", _relPath);
            return rebuild;
        }

        var body = Split(context);
        body.InferenceType();

        if (body.CheckedType is InvalidType)
        {
            DumpIR(body, "InvalidBody");
            throw new InvalidOperationException();
        }

        if (body.Users.Count > 1)
        {
            throw new InvalidOperationException();
        }

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

    public Expr FixInput(FusionBucketContext context, int[][] shapeList)
    {
        var fixedShapeInput = context.Arguments.Zip(shapeList).Select(pair =>
        {
            var (arg, fixedShape) = pair;
            return (Expr)new Call(new FixShape(), arg, fixedShape);
        }).ToArray();
        return ReplaceClone(context.FusionBody, context.Parameters.Zip(fixedShapeInput).ToArray());
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

    private static Expr MakeSlice(FusionBucketContext context, Expr call, Expr originBody)
    {
        var sliceShape = context.SliceShape;
        if (call.CheckedType is TupleType tuple)
        {
            var fields = Enumerable.Range(0, tuple.Count)
                .Select(i => MakeSliceForTensor(sliceShape[i], call[i], context)).ToArray();
            return new IR.Tuple(fields);
        }

        return MakeSliceForTensor(sliceShape, call, context);
    }

    private static Expr MakeSliceForTensor(Expr sliceShape, Expr call, FusionBucketContext context)
    {
        var rank = call.CheckedShape.Rank;
        var simplifyCall = CompilerServices.Rewrite(
            call,
            new IRewriteRule[]
            {
                new FoldStackGetItem(),
                new FoldShapeOf(),
                new FoldTwoReshapes(),
                new FoldTwoCasts(),
                new FoldTwoSlices(),
                new FoldNopBinary(),
                new FoldNopCast(),
                new Neutral.FoldConstCall(),
                new FoldNopReshape(),
                new FoldNopSlice(),
                new FoldIf(),
                new FoldBroadcastShape(),
            },
            new());

        var axes = Tensor.From(Enumerable.Range(0, rank).Select(x => (long)x).ToArray());
        var strides = Tensor.FromScalar(1L, rank);
        var body = (Expr)Slice(simplifyCall, Enumerable.Repeat(0L, rank).ToArray(), Cast(sliceShape, DataTypes.Int64), axes, strides);
        return body;
    }

    private static bool IsFixed(int totalCount, int[][] minFixedShapeList, int[][] maxFixedShapeList) =>
        totalCount == 0 || (minFixedShapeList[0].SequenceEqual(maxFixedShapeList[0]) &&
                            minFixedShapeList[1].SequenceEqual(maxFixedShapeList[1]));

    private static bool ShouldRestore(Call outerCall, BucketFusion fusion)
    {
        if (CallValidator.IsSimple(fusion))
        {
            return true;
        }

        if (outerCall.CheckedType is TupleType tt)
        {
            if (tt.Fields.All(f => f is TensorType t && t.Shape.Rank < 2))
            {
                return true;
            }
        }

        if (outerCall.Arguments.ToArray().Any(arg =>
                arg.CheckedType is TupleType))
        {
            return true;
        }

        return false;
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

    private static bool ShouldBeRebuild(FusionBucketContext context)
    {
        var varInfo = context.DimVarValue(0);
        var entry = MakeSplitEntry(context, varInfo, 0, false, false);
        return entry switch
        {
            IR.Tuple tuple => tuple.Fields.ToArray().Any(ShouldBeRebuild),
            Call => ShouldBeRebuild(entry),
            _ => DumpError(entry),
        };
    }

    private static bool DumpError(Expr entry)
    {
        DumpIR(entry, "FailedEntry");
        throw new InvalidOperationException();
    }

    private static bool ShouldBeRebuild(Expr entry)
    {
        if (entry is Call { Target: IR.Tensors.Slice } c)
        {
            var body = c.Arguments[IR.Tensors.Slice.Input.Index];
            if (body.CheckedShape.IsFixed)
            {
                var visitor = new DynamicCheckVisitor();
                visitor.Visit(body);
                return visitor.HasDynamic;
            }
        }

        return true;
    }

    public class DynamicCheckVisitor : ExprVisitor<Expr, Unit>
    {
        private bool _hasDynamic;

        public bool HasDynamic => _hasDynamic;

        protected override Expr DefaultVisitLeaf(Expr expr) => expr;

        protected override Expr VisitLeafCall(Call expr)
        {
            if (CallValidator.ForceConvert.Contains(expr.Target.GetType().TypeHandle))
            {
                if (!expr.CheckedShape.IsFixed)
                {
                    _hasDynamic = true;
                }
            }

            return expr;
        }
    }
}

internal record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);

public class FullBucket : FunctionPass
{
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext ctx)
    {
        if (!SingleDimVar(CompileSession.CompileOptions.ShapeBucketOptions))
        {
            throw new NotImplementedException("Not Implement multi DimVar for FullBucket");
        }

        var main = (Function)input;
        var replaceItem = main.Parameters.ToArray().Select(param => (param, (Expr)new Var(param.CheckedType))).ToArray();
        var cloneMain = (Function)ReplaceClone(main, replaceItem);
        var options = CompileSession.CompileOptions.ShapeBucketOptions;
        var tmpFusion = new BucketFusion("stackvm", cloneMain.Body, cloneMain.Parameters, Array.Empty<Var>());
        var call = new Call(tmpFusion, main.Parameters.ToArray());
        var dimVarValues = MakeVarValuesForAllSegment(options);
        var list = InputConfList(dimVarValues);
        var shapeData = MakeShapeData(list, options);

        var context = new FusionBucketContext(call, tmpFusion, options, new ShapeExprCache(options.VarMap), 0, shapeData);

        var allFixedShapes = shapeData
            .Select(x =>
                x.InputShapes.Select(iShape => iShape.AsTensor().ToArray<int>().ToArray()).ToArray()).ToArray();
        for (int i = 0; i < shapeData.Length; i++)
        {
            for (int j = 0; j < allFixedShapes.Length; j++)
            {
                context.FixedShapeCache[j] = allFixedShapes[j];
            }
        }

        var newBody = FusionBucket.Split(context);
        foreach (var (oldVar, tmpVar) in replaceItem)
        {
            ReplaceExpr(newBody, tmpVar, oldVar);
        }

        return Task.FromResult((BaseFunction)main.With(body: newBody));
    }

    private static FusionShapeData[] MakeShapeData((Var Key, int Value)[][] list, ShapeBucketOptions options) =>
        list.Select(seg =>
        {
            var varValues = seg.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value));
            var inShape = options.VarMap.Select(pair =>
            {
                var shapeExpr = pair.Key.CheckedShape.IsScalar
                    ? (Expr)Array.Empty<int>()
                    : Stack(new IR.Tuple(pair.Value.Select(x => Cast(x, DataTypes.Int64)).ToArray()), 0);

                var shape = shapeExpr.Evaluate(varValues).AsTensor();
                return shape;
            }).ToArray();
            return new FusionShapeData(Value.None, inShape.Select(Value.FromTensor).ToArray());
        }).ToArray();

    private static (Var Key, int Value)[][] InputConfList(Dictionary<Var, int[]> dimVarValues) =>
        Enumerable.Range(0, dimVarValues.First().Value.Length).Select(i =>
        {
            // 一组里面多个key seg
            return dimVarValues.Select(pair => (pair.Key, Value: pair.Value[i])).ToArray();
        }).ToArray();
}
