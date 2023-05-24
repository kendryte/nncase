using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
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
using static Nncase.PatternMatch.Utility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.Passes.Rules.Neutral.ShapeBucketHelper;
using Dimension = Nncase.IR.Dimension;
using Tuple = System.Tuple;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.Neutral;

public class VarFusion : Fusion
{
    public Var[] EffectVar;

    public VarFusion(string name, string moduleKind, Expr body, ReadOnlySpan<Var> parameters, Var[] effectVar) : base(
        name, moduleKind, body, parameters)
    {
        EffectVar = effectVar;
    }

    public VarFusion(string moduleKind, Expr body, ReadOnlySpan<Var> parameters, Var[] effectVar) : base(moduleKind,
        body,
        parameters)
    {
        EffectVar = effectVar;
    }

    public VarFusion(string name, string moduleKind, Var[] effectVar, Expr body, params Var[] parameters) : base(name,
        moduleKind, body, parameters)
    {
        EffectVar = effectVar;
    }

    public VarFusion(string moduleKind, Var[] effectVar, Expr body, params Var[] parameters) : base(moduleKind, body,
        parameters)
    {
        EffectVar = effectVar;
    }
}

internal static class ShapeBucketHelper
{
    public static Marker[] GetCallInputs(Call call) =>
        call.Arguments.ToArray().OfType<Marker>().Where(x => x.Target is not TensorConst).ToArray();
}

[RuleGenerator]
public partial class MarkerCallToFusion<T> : RewriteRule<Pattern> where T : Op
{
    public override Pattern Pattern => IsRangeOfMarker(
        "callMarker",
        IsCallWildcard(null, IsOp<T>()),
        IsTensorConst());

    public string ModuleKind = "stackvm";

    public string Name;

    public int Counter = 0;

    public Call CurrentCall;

    public string RelPath => $"{Counter}_{CurrentCall.Target.GetType().Name}";

    virtual public bool Check(Call call) { return true;}

    public Var[] MakeEffectVarArray(params Expr[] args)
    {
        var varMap = CompileSession.CompileOptions.ShapeBucketOptions.VarMap;
        var visitor = new FindVar();
        args.ForEach(arg =>
        {
            DumpScope.Current.DumpIR(arg, "arg", RelPath);
            var argShapeExpr = arg.EvaluateShapeExpr(varMap);
            DumpScope.Current.DumpIR(argShapeExpr, "shapeExpr", RelPath);
            visitor.Visit(argShapeExpr);
        });
        return visitor.Vars.ToHashSet().Except(varMap.Keys.ToHashSet()).ToHashSet().ToArray();
    }

    public Expr GetReplace(Marker callMarker)
    {
        var call = (Call)(callMarker.Target);
        CurrentCall = call;
        DumpScope.Current.DumpIR(callMarker, "origin", RelPath);
        var argsMarker = GetCallInputs(call);
        var args = argsMarker.Select(arg => arg.Target).ToArray();
        var set = MakeEffectVarArray(args);
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
        var f = new VarFusion($"{Name}_{Counter}", ModuleKind, set, body, fusionVars);
        var outerCall = new Call(f, args);
        DumpScope.Current.DumpIR(outerCall, "fusion", RelPath);
        Counter++;
        return outerCall;
    }
}

public class Conv2DToFusion : MarkerCallToFusion<Conv2D> {}

public class Conv2DTransposeToFusion : MarkerCallToFusion<Conv2DTranspose>
{
    // when OutputShape is Const, it means output shape is not effected by input.
    public override bool Check(Call call) => call.Arguments[Conv2DTranspose.OutputShape.Index] is not Const;
}

public class MatmulToFusion : MarkerCallToFusion<MatMul> {}

[RuleGenerator]
public partial class ReplaceRewrite : RewriteRule<Pattern>
{
    public ReplaceRewrite(Var[] vars, Dictionary<Var, Expr[]> fusionInputData, Expr[] fusionInput,
        Dictionary<Var, Expr[]> fusionInputsShape)
    {
        var options = CompileSession.CompileOptions.ShapeBucketOptions;
        dict = options.RangeInfo;
        InputInfo = options.VarMap;
        segmentCount = options.SegmentsCount;
        DimVar = vars;
        FusionInputData = fusionInputData;
        FusionInputs = fusionInput;
        FusionInputsShape = fusionInputsShape;
    }

    private int segmentCount;
    private Dictionary<Var, Expr[]> InputInfo;
    private Dictionary<Var, Expr[]> FusionInputData;
    private Dictionary<Var, Expr[]> FusionInputsShape;
    private Dictionary<string, (int, int)> dict;
    private Expr[] FusionInputs;

    public override Pattern Pattern => IsRangeOfMarker("callMarker",
        IsCallWildcard("call", IsAlt(IsOp<MatMul>(), IsOp<Conv2D>(), IsOp<Conv2DTranspose>())), IsTensorConst());

    private Var[] DimVar;

    public record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);

    public Expr FixInput(Call call, Marker[] callInputs, int[][] fixedShapeList, Marker callMarker)
    {
        var arguments = callInputs.Zip(fixedShapeList)
            .Select(pair =>
            {
                var (marker, fixedShape) = pair;
                return marker.With(target: new Call(new FixShape(), marker.Target, fixedShape));
            }).ToArray();
        var newArgs = ReplaceUtility.ReplaceItems(call.Arguments.ToArray(),
            arguments.Select((x, i) => (i, (Expr)x)).ToArray());
        return callMarker.With(target: call.With(arguments: newArgs));
    }

    public Expr? GetReplace(Marker callMarker, Call call)
    {
        Console.WriteLine($"Rewrite{count}");
        var (minDict, maxDict) = GetBoundDict(InputInfo, dict);
        PrintBoundDict(minDict, maxDict);
        CheckAlive();

        var callInputs = GetCallInputs(call);
        var minFixedShapeList = ComputeFixedShape(callInputs, minDict);
        var maxFixedShapeList = ComputeFixedShape(callInputs, maxDict);

        // 2. get dim info(inputIndex, (dimIndex, range)
        var counts = ComputeCounts(minFixedShapeList, maxFixedShapeList, out int totalCount);
        PrintMatmulInfo(call, minFixedShapeList, maxFixedShapeList, totalCount);
        PrintSegmentInfo(counts);
        if (totalCount == 0 || (minFixedShapeList[0].SequenceEqual(maxFixedShapeList[0]) &&
                                minFixedShapeList[1].SequenceEqual(maxFixedShapeList[1])))
        {
            var fix = FixInput(call, callInputs, minFixedShapeList, callMarker);
            return fix;
        }

        var varValues = MakeVarValuesForAllSegment(segmentCount);
        var info = ComputeSegmentInfo(counts, segmentCount);
        var expr = Split(call, callMarker, callInputs, info, 0, 1, varValues);
        expr.InferenceType();
        // DumpScope.Current.DumpIR(expr, "SplitResult", $"{count}");
        count++;
        return expr;
    }

    private static (int inputIndex, (int First, (int First, int Second) Second)[] range)[] ComputeCounts(
        int[][] minFixedShapeList, int[][] maxFixedShapeList, out int totalCount)
    {
        var counts = minFixedShapeList.Zip(maxFixedShapeList).Select((pair, inputIndex) =>
        {
            // Console.WriteLine(inputIndex);
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

    private void CheckAlive()
    {
        foreach (var value in FusionInputData.Values)
        {
            foreach (var ex in value)
            {
                if (!ex.IsAlive)
                {
                    throw new NotImplementedException();
                }
            }
        }
    }

    private static void PrintBoundDict(Dictionary<Var, IValue> minDict, Dictionary<Var, IValue> maxDict)
    {
        Console.WriteLine("MinDict");
        foreach (var (key, value) in minDict)
        {
            Console.WriteLine($"{key.Name} {string.Join(",", value.AsTensor().ToArray<int>())}");
        }

        Console.WriteLine("MaxDict");
        foreach (var (key, value) in maxDict)
        {
            Console.WriteLine($"{key.Name} {string.Join(",", value.AsTensor().ToArray<int>())}");
        }
    }

    private static SegmentInfo ComputeSegmentInfo(
        (int inputIndex, (int First, (int First, int Second) Second)[] range)[] counts, int segmentCount)
    {
        var (iIndex, dimIndex, (min, max)) = counts.Select(x =>
        {
            Debug.Assert(x.range.Length <= 2);
            return (x.inputIndex, x.range[0].First, x.range[0].Second);
        }).ToArray().First();

        var segments = ComputeSegmentList(segmentCount, min, max);
        var info = new SegmentInfo(iIndex, dimIndex, segments);
        return info;
    }

    private static void PrintSegmentInfo((int inputIndex, (int First, (int First, int Second) Second)[] range)[] counts)
    {
        Console.Write("InputSegement ");
        foreach ((int inputIndex, var range) in counts)
        {
            Console.Write($"{inputIndex} {string.Join(", ", range)}");
        }

        Console.WriteLine();
    }

    private Dictionary<Var, int[]> MakeVarValuesForAllSegment(int segmentCount)
    {
        var varAndInputAllSegment = dict.ToDictionary(pair => pair.Key, pair =>
        {
            var (min, max) = pair.Value;
            var segments = ComputeSegmentList(segmentCount, min, max);
            // Console.WriteLine("segments list");
            // Console.WriteLine($"{min}, {max}");
            // Console.WriteLine($"{((max - min) / segmentCount)}");
            // Console.WriteLine(string.Join(",", segments));
            return segments;
        });

        var vars = InputInfo.Values.SelectMany(x => x).OfType<Var>().ToHashSet().ToArray();
        // DimVarName -> Dict.key -> Dict.Value
        var varValues = varAndInputAllSegment.ToDictionary(pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => { return pair.Value.OrderByDescending(x => x).ToArray(); });
        return varValues;
    }

    public static int[] ComputeSegmentList(int segmentCount, int min, int max)
    {
        var size = (max - min) / segmentCount;
        return Enumerable.Range(0, segmentCount - 1).Select(i => min + i * size).Append(max).ToArray();
    }

    private void PrintMatmulInfo(Call call, int[][] minFixedShapeList, int[][] maxFixedShapeList, int totalCount) =>
        Console.WriteLine("min MatmulShape:" +
                          string.Join("|", minFixedShapeList.Select(s => DumpUtility.SerializeShape(s))) +
                          $" {call.Metadata.OutputNames}" + " max MatmulShape:" +
                          string.Join("|", maxFixedShapeList.Select(s => DumpUtility.SerializeShape(s))) +
                          $" {call.Metadata.OutputNames}. count:{string.Join(",", totalCount)} var:{DimVar.Count()}");

    public static (Dictionary<Var, IValue> MinDict, Dictionary<Var, IValue> MaxDict) GetBoundDict(
        Dictionary<Var, Expr[]> inputInfo, Dictionary<string, (int Min, int Max)> limitDict)
    {
        // find vars in Input ShapeExpr
        var vars = inputInfo.Values.SelectMany(x => x).OfType<Var>().ToHashSet().ToArray();

        // DimVarName -> Dict.key -> Dict.Value
        var minDict = limitDict.ToDictionary(pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => (IValue)Value.FromTensor(pair.Value.Min));
        var maxDict = limitDict.ToDictionary(pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => (IValue)Value.FromTensor(pair.Value.Max));
        return (minDict, maxDict);
    }

    private int[][] ComputeFixedShape(Marker[] callInputs, Dictionary<Var, IValue> varInfo) =>
        callInputs.Select((arg, i) =>
        {
            var fixedShape = ShapeEvaluate(arg, InputInfo, varInfo, FusionInputData, i);
            return fixedShape;
        }).ToArray();


    private Expr Split(Call call, Marker callMarker, Marker[] callInputs, SegmentInfo info, int current,
        int limit, Dictionary<Var, int[]> varValues)
    {
        FusionInputs = callInputs.Zip(FusionInputs).Select(pair => pair.Item1.With(target: pair.Item2)).ToArray();

        // 分段是针对input做的，而不是替换了input。
        // arg var -> compute
        // arg var -> bucket -> compute
        // arg -> bucket -> compute
        var (inputIndex, dimIndex, segments) = info;
        var sp = ConstantOfShape(new[] { 1 }, Cast(0, call.CheckedDataType));
        int i = 0;

        // todo: fusionInputs
        var body = segments.OrderByDescending(x => x).Aggregate(
            (Expr)IR.F.Math.Require(false, sp, "input dim large than limit"),
            (sum, seg) =>
            {
                // Console.WriteLine("segment value");
                // Console.WriteLine(seg);
                var cond = Cast(ShapeOf(FusionInputs[inputIndex]), DataTypes.Int32)[dimIndex] <= seg;
                // select var value for current segment
                DumpScope.Current.DumpIR(call, $"call_before{i}");
                var varInfo = varValues.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value[i]));
                var thenBody = current + 1 < limit
                    ? Split(call, callMarker, callInputs, info, current + 1, limit, varValues)
                    : MakeSplitEntry(call, callMarker, callInputs, InputInfo, varInfo, FusionInputData,
                        FusionInputsShape, FusionInputs);
                DumpScope.Current.DumpIR(call, $"after_{i}");
                var elseBody = sum;
                i++;
                var result = new If(cond, thenBody, elseBody);
                DumpScope.Current.DumpIR(result, $"{i}");
                return result;
            });
        // let args
        if (body is If @if)
        {
            return @if.With(paramList: FusionInputs);
        }
        return body;
    }

    // todo: 问题本质是每个分支需要其单独的var，因为其参数需要fixed的input
    public static Expr MakeSplitEntry(Expr originCall, Marker callMaker, Marker[] callInputs,
        Dictionary<Var, Expr[]> inputInfo,
        Dictionary<Var, IValue> varInfo, Dictionary<Var, Expr[]> fusionInputdata,
        Dictionary<Var, Expr[]> fusionInputsShape, Expr[] fusionInputs)
    {
        var call = originCall.Clone();

        Console.WriteLine("MakeSplitEntry");
        // var callInputs = GetCallInputs((Call)call);
        var newInputs = GetCallInputs((Call)call).Select(x => (Var)x.Target).ToArray();

        var fixInputs = callInputs
            .Select((arg, i) => PreProcess(arg, inputInfo, varInfo, fusionInputdata, fusionInputs, i)).ToArray();
        DumpScope.Current.DumpIR(call, "CopyBeforeReplace");

        // var v1 = varAndFixArg[0].OriginVar;
        var v2 = fusionInputsShape.Keys.First();
        // call参数里面的var
        // Console.WriteLine(varAndFixArg[0].OriginVar.GetHashCode());
        call = newInputs.Zip(fixInputs).Aggregate(call, (sum, pair) =>
        {
            return ReplaceUtility.ReplaceExpr(sum, pair.Item1, pair.Item2);
        });
        call = callMaker.With(target: call);
        DumpScope.Current.DumpIR(call, "AfterReplace");


        // 如何创建一个new function
        // var oldVars = varAndFixArg.Select(pair => pair.OriginVar).ToArray();
        // // var functionVar = oldVars.Select(oldVar => oldVar.Clone()).ToArray();
        // var inputsShape = oldVars.Select(v =>
        // {
        //     // find copy
        //     var testV = fusionInputsShape.FindFirst(oldV => oldV.Key.GlobalVarIndex == v.GlobalVarIndex);
        //     return (v, testV.Value);
        // }).ToDictionary(pair => pair.v, pair => pair.Value);


        // shape表达式应该是关于原始input的，所以shape表达式不受影响，之后安心替换var就可以了把
        var originShape = originCall.EvaluateShapeExpr(fusionInputsShape);
        originShape.InferenceType();

        var rank = call.CheckedShape.Rank;
        var body = (Expr)Slice(call, Enumerable.Repeat(0, rank).ToArray(), Cast(originShape, DataTypes.Int32), rank);
        // var body = call;

        // var oldVars = callInputs.Select(x => (Var)x.Target).ToArray();
        // oldVars.Zip(fusionInputs).Aggregate(body, (sum, tuple) =>
        // {
            // return ReplaceUtility.ReplaceExpr(sum, tuple.Item1, tuple.Item2)
        // });

        return body;

        // 传进来的是clone的call，但是这样对var找不到对应的var
        // 但是针对非clone的call调用replace的话，那么会修改外部的东西，因为现在的replace是mutable的

        // 现在这里必须要保证var不变的情况下创建一个新的call拿进来用才行


        // 1. 所有input进行前处理。

        // todo：是不是应该对整个var处理啊...而不是针对call，最后将var替换，但是替换var的时候也是有一样的问题
        // 每个if的分支里面其实应该是一套单独的东西，要不还是将if分支里面的东西转成一个function call...但是好像还是不能避免


        // Console.WriteLine("MakeEntry");
        // var varAndFixArg = callInputs
            // .Select((arg, i) => PreProcess(arg, inputInfo, varInfo, fusionInputdata, fusionInputs, i)).ToArray();
        // var fixedShapeList = varAndFixArg.Select(x => x.Item2).ToArray();
        // var markers = callInputs;
        // var bucketInputPairs = fixedShapeList.Select((arg, i) => (i, (Expr)markers[i].With(target: arg)));

        // var normalInputPairsList =
        //     call.Arguments.ToArray().Select((arg, i) => (i, arg)).Where(pair => pair.arg is Call);
        // var normalInputPairs = normalInputPairsList.Select(pair =>
        // {
        //     var (arg, i) = pair;
        //     var newArg = varAndFixArg.Aggregate((Expr)arg, (sum, varAndArg) =>
        //     {
        //         DumpScope.Current.DumpIR(sum, "body", $"{count}");
        //         DumpScope.Current.DumpIR(varAndArg.Item1, "target", $"{count}");
        //         DumpScope.Current.DumpIR(varAndArg.Item2, "expr", $"{count}");
        //         var newExpr = ReplaceUtility.ReplaceExpr(sum, varAndArg.Item1, varAndArg.Item2);
        //         if (newExpr == sum)
        //         {
        //             Console.WriteLine();
        //         }
        //
        //         return newExpr;
        //     });
        //     DumpScope.Current.DumpIR(arg, "after_arg", $"{count}");
        //     return (i, newArg);
        // });
        // var args = ReplaceUtility.ReplaceItems(call.Arguments.ToArray(), bucketInputPairs.ToArray());
        // var then = (Expr)call.With(arguments: args);
        // var thenWithMarker = callMaker.With(target: then);
        // thenWithMarker.InferenceType();
        // DumpScope.Current.DumpIR(thenWithMarker, "thenWithMarker", $"{count}");
        // // Console.WriteLine($"PostShape: {DumpUtility.SerializeShape(thenWithMarker.CheckedShape.ToValueArray())}");



        // get preprocess input
        // var varAndFixArg = callInputs
            // .Select((arg, i) => PreProcess(arg, inputInfo, varInfo, fusionInputdata, fusionInputs, i)).ToArray();
        // var fixedShapeList = varAndFixArg.Select(x => x.Item2).ToArray();
        // var bucketInputPairs = fixedShapeList.Select((arg, i) => (i, (Expr)callInputs[i].With(target: arg)));
        // 问题是这里替换var的时候，没有把argument里的var也换掉，但是也不能直接替换，因为是mutable的



        // var originCall = call;
        var then = call;
        // var then = varAndFixArg.Aggregate((Expr)call, (sum, pair) =>
        // {
        //     return ReplaceUtility.ReplaceExpr(sum, pair.Item1, pair.Item2);
        // });
        // // var pairs = tmpInputs.Select((tmpVar, i) => (i, (Expr)callInputs[i].With(target: tmpVar))).ToArray();
        //

        // generate a new call
        // var args = ReplaceUtility.ReplaceItems(call.Arguments.ToArray(), pairs);
        // var then = (Expr)call.With(arguments: args);
        var thenWithMarker = callMaker.With(target: then);

        // replace var with fixedInputs
        // 因为所有的argument都要用FixInput来计算
        // var replacePair = tmpInputs.Zip(varAndFixArg).Select(tup => (tup.Item1, tup.Item2.FixInput)).ToArray();
        // DumpScope.Current.DumpIR(thenWithMarker, "thenWithMarker");
        // DumpScope.Current.DumpIR(thenWithMarker.Clone(), "thenWithMarkerClone");
        // var body = replacePair.Aggregate((Expr)thenWithMarker, (sum, varAndArg) =>
        // {
        //     return ReplaceUtility.ReplaceExpr(sum, varAndArg.Item1, varAndArg.Item2);
        // });
        // if (body == thenWithMarker)
        // {
        //     Console.WriteLine();
        // }

        // 现在clone会发生什么，var是不是也会变成另一个呢

        // post process call
        // post中shape表达式的var不能发生改变，因为要针对原始的arg计算
        // DumpScope.Current.DumpIR(originCall, "origin");
        // DumpScope.Current.DumpIR(thenWithMarker, "thenWithMarker");
        // var originShape = originCall.EvaluateShapeExpr(fusionInputsShape);
        // originShape.InferenceType();
        // var rank = call.CheckedShape.Rank;
        // var post = (Expr)Slice(thenWithMarker, Enumerable.Repeat(0, rank).ToArray(), Cast(originShape, DataTypes.Int32), rank);
        // post.InferenceType();








        // preprocess本质是针对原始的arg，但是目前能获取到的只有var，
        // post针对的是输出，但是输出的时候var必须存在于计算的参数中，因为需要对原始的var做shape of才行

        // 在post后，甚至整个分段结束再将var替换为arg。因为post里面的shape表达式会引用到原始的inputs，不论哪个if分支都是一样的
        //
        // 也就是说一开始不实际替换，只需要使用args构建出对应的bucket input就可以，最后把var和整个input进行替换。但要注意
        // slice的是fix的call


        // DumpScope.Current.DumpIR(post, "PostExpr");
        // return post;
    }

    // 将一个var使用fusion的arg代替，而这个arg则是经过pad过的
    // 返回的是这个var以及对应pad后的arg，用于其他非marker的参数替换这个var
    public static Expr PreProcess(Marker input, Dictionary<Var, Expr[]> inputInfo,
        Dictionary<Var, IValue> varValues, Dictionary<Var, Expr[]> fusionInputData, Expr[] fusionInputs, int i)
    {
        var fixedShape = ShapeEvaluate(input, inputInfo, varValues, fusionInputData);
        // return new Call(new BucketPad(), input, fixedShape);
        var pads = fixedShape - Cast(ShapeOf(fusionInputs[i]), DataTypes.Int32);
        var paddings = Transpose(Stack(new IR.Tuple(Enumerable.Repeat(0, fixedShape.Length).ToArray(), pads), 0),
            new[] { 1, 0 });
        var fixedInput = IR.F.NN.Pad(fusionInputs[i], paddings, PadMode.Constant, Cast(0, input.CheckedDataType));
        var fixedResult = new Call(new FixShape(), fixedInput, fixedShape);
        return fixedResult;
    }

    private static int count = 0;

    // info:(InputVar -> DimVar)
    // VarInfo:(DimVar -> Value)
    // fusionInfo:(InputVar -> DimVar)
    public static int[] ShapeEvaluate(Expr expr, Dictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo,
        Dictionary<Var, Expr[]> fusionInfo, int i = 0)
    {
        // var info is used for compute shape expr
        var dummyInput = MakeDummyInput(info, varInfo);
        var fusionDummyInput =
            MakeDummyInput(fusionInfo,
                varInfo.Concat(dummyInput).ToDictionary(pair => pair.Key, pair => pair.Value));
        // Console.WriteLine("dummyInput");

        // Console.WriteLine("new info");
        // DumpScope.Current.DumpIR(expr, "ShapeEval382", $"{count}");
        foreach (var (key, value) in fusionInfo)
        {
            for (var i1 = 0; i1 < value.Length; i1++)
            {
                // DumpScope.Current.DumpIR(value[i], $"{i}", $"{count}/" + key.Name);
            }
        }

        var shapeExpr =
            expr.EvaluateShapeExpr(info.Concat(fusionInfo).ToDictionary(pair => pair.Key, pair => pair.Value));
        shapeExpr.InferenceType();

        // used for shape expr evaluate
        // 1. main input
        // 2. fusion input
        // 3. shape var
        var newEvaluatorInfo = dummyInput.Concat(fusionDummyInput).Concat(varInfo)
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        foreach (var (key, value) in newEvaluatorInfo)
        {
            // Console.WriteLine(key.Name);
            var arr = value.AsTensor().ToArray<int>();
            if (arr.Length < 20)
            {
                // Console.WriteLine(string.Join(",", arr));
            }
            else
            {
                // Console.WriteLine(DumpUtility.SerializeShape(value.AsTensor().Shape));
            }
        }

        // DumpScope.Current.DumpIR(shapeExpr, "ShapeExpr", $"{count}");
        // Console.WriteLine("evaluate shapeExpr");
        // todo:this error
        // todo: shape expr inference
        var shape = shapeExpr.Evaluate(newEvaluatorInfo);
        // Console.WriteLine("after evaluate shapeExpr");

        return shape.AsTensor().ToArray<int>();
    }

    // make dummy value from InputInfo
    // VarInfo:(DimVar -> Value)
    private static Dictionary<Var, IValue>
        MakeDummyInput(Dictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo) =>
        info.ToDictionary(pair => pair.Key,
            pair =>
            {
                try
                {
                    var shapeExpr = Stack(new IR.Tuple(pair.Value.Select(x => Cast(x, DataTypes.Int32)).ToArray()), 0);
                    DumpScope.Current.DumpIR(shapeExpr, "mkInputShapeExpr", $"{count}");
                    var shape = shapeExpr.Evaluate(varInfo).AsTensor();
                    return ConstantOfShape(
                        shape,
                        Cast(0, pair.Key.CheckedDataType)).Evaluate(varInfo);
                }
                catch (Exception e)
                {
                    // Console.WriteLine(e);
                    throw;
                }
            });

    public static Call PostProcess(Expr call, Expr originCall, Dictionary<Var, Expr[]> fusionInputsShape)
    {
        if (!call.InferenceType())
        {
            DumpScope.Current.DumpIR(call, "invalidPostProcess", count.ToString());
        }

        var shape = originCall.EvaluateShapeExpr(fusionInputsShape);
        shape.InferenceType();
        // DumpScope.Current.DumpIR(shape, "PostShapeExpr", count.ToString());
        var rank = call.CheckedShape.Rank;
        return Slice(call, Enumerable.Repeat(0, rank).ToArray(), Cast(shape, DataTypes.Int32), rank);
    }
}

[RuleGenerator]
public partial class FusionBucket : RewriteRule<Pattern>
{
    private static int counter = 0;

    public override Pattern Pattern => IsCall("call",
        IsFusion("fusion", "stackvm", IsWildcard(), GenerateParameters(null)),
        GenerateParameters(null));

    public Expr? GetReplace(Call call, VarFusion fusion)
    {
        Console.WriteLine($"FusionBucketGetReplace {counter}");
        var relPath = $"ShapeBucketDebug/{counter}";
        var ctx = new RunPassContext();
        var varMap = CompileSession.CompileOptions.ShapeBucketOptions.VarMap;
        // fusion -> 关联到外部的表达式，用于计算实际的shape范围
        var fusionInputInfo = MakeFusionInputShapeInfo(call, fusion, varMap);
        CheckAlive(fusionInputInfo);
        // ensure alive in rewrite, release when return
        using var _ = new ExprPinner(fusionInputInfo.Values.SelectMany(x => x).ToArray());
        var oldBody = fusion.Body;
        var args = call.Arguments.ToArray();
        // fusion 内部var的表达式，
        var fusionInputShapes = GetFusionInputShapes(fusion, args);

        // DumpScope.Current.DumpIR(fusion.Body, "oldBody", relPath);
        // PrintMinMaxShape("oldBody", call, oldBody, relPath, fusionInputInfo);
        var newBody = CompilerServices.Rewrite(fusion.Body,
            new[]
            {
                new ReplaceRewrite(fusion.EffectVar, fusionInputInfo, args,
                    fusionInputShapes)
            }, ctx);
        if (oldBody == newBody)
        {
            return null;
        }

        // DumpScope.Current.DumpIR(newBody, "BeforeReplace", relPath);
        // FixInput Replace Var
        newBody = ReplaceFusionVarWithCallArgs(fusion, args, newBody);
        DumpScope.Current.DumpIR(newBody, "BucketResult", relPath);
        counter++;
        return newBody;
    }

    private static Expr ReplaceFusionVarWithCallArgs(VarFusion fusion, Expr[] args, Expr newBody) =>
        fusion.Parameters.ToArray().Zip(args).Aggregate(newBody, (sum, pair) =>
        {
            var (param, arg) = pair;
            return ReplaceExpr(sum, param, arg);
        });

    private static Dictionary<Var, Expr[]> GetFusionInputShapes(VarFusion fusion, Expr[] args)
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

    private void PrintMinMaxShape(string prefix, Call call, Expr body, string relPath,
        Dictionary<Var, Expr[]> fusionInputInfo)
    {
        // // Console.WriteLine(prefix);
        // var (lhsMin, lhsMax) = ReplaceRewrite.GetBoundDict(ModelInputsVarShape, DimVarRange);
        // var lhsMinShape = ReplaceRewrite.ShapeEvaluate(call.Arguments[0], ModelInputsVarShape, lhsMin, fusionInputInfo);
        // var lhsMaxShape = ReplaceRewrite.ShapeEvaluate(call.Arguments[0], ModelInputsVarShape, lhsMax, fusionInputInfo);
        //
        // var (rhsMin, rhsMax) = ReplaceRewrite.GetBoundDict(ModelInputsVarShape, DimVarRange);
        // var rhsMinShape = ReplaceRewrite.ShapeEvaluate(call.Arguments[1], ModelInputsVarShape, rhsMin, fusionInputInfo);
        // var rhsMaxShape = ReplaceRewrite.ShapeEvaluate(call.Arguments[1], ModelInputsVarShape, rhsMax, fusionInputInfo);
        //
        //
        // var (min, max) = ReplaceRewrite.GetBoundDict(ModelInputsVarShape, DimVarRange);
        // var minShape = ReplaceRewrite.ShapeEvaluate(body, ModelInputsVarShape, min, fusionInputInfo);
        // // Console.WriteLine("Start BodyShapeExpr Max");
        // var maxShape = ReplaceRewrite.ShapeEvaluate(body, ModelInputsVarShape, max, fusionInputInfo);
        //

        // don't have fusion input info
        // var bodyShapeExpr = body.EvaluateShapeExpr(InputInfo);
        // // DumpScope.Current.DumpIR(bodyShapeExpr, "bodyShapeExpr", relPath);


        // var minShape = shapeExpr.Evaluate(min).AsTensor().ToArray<int>();
        // var maxShape = shapeExpr.Evaluate(max).AsTensor().ToArray<int>();
        // Console.WriteLine(
        // $"{prefix}_{relPath} lhs: {DumpUtility.SerializeShape(lhsMinShape)} | {DumpUtility.SerializeShape(lhsMaxShape)} rhs: {DumpUtility.SerializeShape(rhsMinShape)} | {DumpUtility.SerializeShape(rhsMaxShape)}  bodyShapeExpr {DumpUtility.SerializeShape(minShape)} | {DumpUtility.SerializeShape(maxShape)}");
    }

    private Dictionary<Var, Expr[]> MakeFusionInputShapeInfo(Call call, VarFusion fusion,
        Dictionary<Var, Expr[]> varMap)
    {
        var data = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray().Select((arg, i) =>
        {
            // Console.WriteLine("arg shape eval");
            var result = arg.EvaluateShapeExpr(varMap);
            // Console.WriteLine("Before Infer");
            result.InferenceType();
            // Console.WriteLine("Infer ok");
            // DumpScope.Current.DumpIR(arg, $"fusionInput_{i++}", $"ShapeBucketDebug/{counter}");
            // DumpScope.Current.DumpIR(result, $"fusionInput_{i++}_shapeExpr", $"ShapeBucketDebug/{counter}");
            return Enumerable.Range(0, arg.CheckedShape.Rank).Select(i => result[i]).ToArray();
        })).Select(pair => new KeyValuePair<Var, Expr[]>(pair.First, pair.Second));
        var fusionInputData = data.ToDictionary(pair => pair.Key, pair => pair.Value);
        return fusionInputData;
    }
}

public class FindVar : ExprVisitor<Expr, Unit>
{
    public HashSet<Var> Vars = new();

    protected override Expr VisitLeafVar(Var expr)
    {
        Vars.Add(expr);
        return expr;
    }

    protected override Expr DefaultVisitLeaf(Expr expr) => expr;
}

