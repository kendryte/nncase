using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using DryIoc.ImTools;
using Microsoft.Toolkit.HighPerformance;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes.Transforms;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.PatternMatch.F.Math;

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

[RuleGenerator]
public partial class MatmulToFusion : IRewriteRule
{
    private static int counter = 0;

    // public IPattern Pattern => IsMatMul(
    //     IsWildcard("arg0"),
    //     IsWildcard("arg1"));
    public IPattern Pattern => IsRangeOfMarker("matmul", IsMatMul(
        IsRangeOfMarker("arg0", IsWildcard(), IsTensorConst()),
        IsRangeOfMarker("arg1", IsWildcard(), IsTensorConst())), IsTensorConst());

    public MatmulToFusion(Dictionary<Var, Expr[]> varMap)
    {
        VarMap = varMap;
    }

    private Dictionary<Var, Expr[]> VarMap;

    public Expr? GetReplace(Marker arg0, Marker arg1, Marker matmul)
    {
        var path = $"{counter}";
        // DumpScope.Current.DumpIR(matmul, "origin", path);
        // if (counter > 3)
        // {
            // return null;
        // }
        // Console.WriteLine($"MatMulToFusion count {counter++}");
        // DumpScope.Current.DumpIR(arg0, "arg0", path);
        // DumpScope.Current.DumpIR(arg1, "arg1", path);
        var visitor = new FindVar();
        // Console.WriteLine("argExpr0");
        var arg0Expr = arg0.EvaluateShapeExpr(VarMap);
        // Console.WriteLine("argExpr1");
        var arg1Expr = arg1.EvaluateShapeExpr(VarMap);
        // Console.WriteLine("visit");
        visitor.Visit(arg0Expr);
        visitor.Visit(arg1Expr);
        var set = visitor.Vars.ToHashSet().Except(VarMap.Keys.ToHashSet()).ToHashSet();
        var lhs = new Var(arg0.CheckedType);
        var rhs = new Var(arg1.CheckedType);
        var inputLhs = arg0.With(target: lhs);
        var inputRhs = arg1.With(target: rhs);
        var m = IR.F.Math.MatMul(inputLhs, inputRhs).InheritMetaData(matmul.Target);
        var matmulWithMarker = matmul.With(target: m);
        // Debug.Assert(set.Count <= 1);
        // Console.WriteLine($"matmul_{counter} GetReplace Set {string.Join(",", set.Select(x => x.Name).ToArray())}");
        // foreach (var var in set)
        // {
        // // Console.WriteLine(var.Name);
        // }

        // // Console.WriteLine("GetReplace End");

        var f = new VarFusion($"matmul_{counter++}", "stackvm", set.ToArray(), matmulWithMarker, lhs, rhs);
        var c = new Call(f, arg0, arg1);
        // DumpScope.Current.DumpIR(inputLhs, "lastArg0", path);
        // DumpScope.Current.DumpIR(inputRhs, "lastArg1", path);
        // DumpScope.Current.DumpIR(c, "MatmulToFusionResult", path);
        return c;
    }
}

[RuleGenerator]
public partial class ReplaceRewrite : IRewriteRule
{
    public ReplaceRewrite(Var[] var, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, Expr[]> fusionInputData,
        Dictionary<string, (int, int)> dict, Expr[] fusionInput, Dictionary<Var, Expr[]> fusionInputsShape,
        int segmentCount)
    {
        DimVar = var;
        InputInfo = inputInfo;
        FusionInputData = fusionInputData;
        this.dict = dict;
        FusionInputs = fusionInput;
        FusionInputsShape = fusionInputsShape;
        this.segmentCount = segmentCount;
    }

    private int segmentCount;
    private Dictionary<Var, Expr[]> InputInfo;
    private Dictionary<Var, Expr[]> FusionInputData;
    private Dictionary<Var, Expr[]> FusionInputsShape;
    private Dictionary<string, (int, int)> dict;
    private Expr[] FusionInputs;

    public IPattern Pattern => IsRangeOfMarker("marker", IsMatMul(null, "call",
            IsRangeOfMarker("lhsMarker", IsWildcard(), IsWildcard()),
            IsRangeOfMarker("rhsMarker", IsWildcard(), IsWildcard())),
        IsTensorConst());

    private Var[] DimVar;

    public record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);

    // public Call FixInput(Call call, int[][] fixedShapeList, Marker marker)
    // {
    //     var arguments = call.Arguments.ToArray().Zip(fixedShapeList)
    //         .Select(pair => new Call(new FixShape(), pair.First, pair.Second)).ToArray();
    //     return call.With(arguments: arguments);
    // }
    public Expr FixInput(Call call, int[][] fixedShapeList, Marker marker)
    {
        var arguments = call.Arguments.ToArray().Zip(fixedShapeList)
            .Select((pair, i) =>
            {
                var (arg, fixedShape) = pair;
                var marker = (Marker)arg;
                if (FusionInputs[i] is Marker { Target: TensorConst tensorConst })
                {
                    return marker.With(target: tensorConst);
                }

                return marker.With(target: new Call(new FixShape(), marker.Target, fixedShape));
            }).ToArray();
        return marker.With(target: call.With(arguments: arguments));
    }

    // compute segment
    // segments count
    public Expr? GetReplace(Marker marker, Call call, Marker lhsMarker, Marker rhsMarker)
    {
        // Console.WriteLine($"start rewrite {count}");

        var (minDict, maxDict) = GetBoundDict(InputInfo, dict);

        // Console.WriteLine("MinDict");
        foreach (var (key, value) in minDict)
        {
            // Console.WriteLine($"{key.Name} {string.Join(",", value.AsTensor().ToArray<int>())}");
        }

        // Console.WriteLine("MaxDict");
        foreach (var (key, value) in maxDict)
        {
            // Console.WriteLine($"{key.Name} {string.Join(",", value.AsTensor().ToArray<int>())}");
        }

        foreach (var value in FusionInputData.Values)
        {
            foreach (var ex in value)
            {
                if (!ex.IsAlive)
                {
                    // Console.WriteLine("not alive in replace");
                    throw new NotImplementedException();
                }
            }
        }

        // Console.WriteLine("ComputeMinFixedShape");
        // 1. compute dim range
        var minFixedShapeList = ComputeFixedShape(call, minDict);
        // Console.WriteLine("ComputeMaxFixedShape");
        var maxFixedShapeList = ComputeFixedShape(call, maxDict);


        // Console.WriteLine("GetDimInfo");
        // 2. get dim info(inputIndex, (dimIndex, range)
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
        var totalCount = counts.Length;

        PrintMatmulInfo(call, minFixedShapeList, maxFixedShapeList, totalCount);
        // PrintSegmentInfo(counts);
        // Console.WriteLine($"total count {totalCount}");
        if (minFixedShapeList[0].SequenceEqual(maxFixedShapeList[0]) &&
            minFixedShapeList[1].SequenceEqual(maxFixedShapeList[1]))
        {
            // todo: 做了不该做的，写个test验证下
            Console.WriteLine("same, not be process");
            var fix = FixInput(call, minFixedShapeList, marker);
            // DumpScope.Current.DumpIR(fix, $"FixInput_{count++}", "fixInput");
            return fix;
            // return null;
        }

        // todo: no effect, fix this
        if (totalCount == 0)
        {
            Console.WriteLine("total count = 0, not be process");
            return FixInput(call, minFixedShapeList, marker);
            // return null;
        }

        var varValues = MakeVarValuesForAllSegment(segmentCount);
        // 代入所有的segments
        var info = ComputeSegmentInfo(counts, segmentCount);
        var expr = Split(call, marker, lhsMarker, rhsMarker, info, 0, 1, varValues);
        expr.InferenceType();
        // expr = CompilerServices.Rewrite(expr, new IRewriteRule[] { new FoldShapeOf(), new FoldConstCall() }, new());
        // var ok = expr.InferenceType();
        // if (!ok)
        // {
            // throw new InvalidOperationException();
        // }

        // DumpScope.Current.DumpIR(expr, "SplitResult", $"{count}");
        Console.WriteLine("SplitOk");
        count++;
        return expr;
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
        // todo:不能混用string，注意
        // find vars in Input ShapeExpr
        var vars = inputInfo.Values.SelectMany(x => x).OfType<Var>().ToHashSet().ToArray();

        // DimVarName -> Dict.key -> Dict.Value
        var minDict = limitDict.ToDictionary(pair => vars.FindFirst(v => v.Name == pair.Key),
            pair =>
            {
                return (IValue)Value.FromTensor(pair.Value.Min);
            });
        var maxDict = limitDict.ToDictionary(pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => (IValue)Value.FromTensor(pair.Value.Max));
        return (minDict, maxDict);
    }

    private int[][] ComputeFixedShape(Call call, Dictionary<Var, IValue> varInfo) =>
        call.Arguments.ToArray().Select((arg, i) =>
        {
            var fixedShape = ShapeEvaluate(arg, InputInfo, varInfo, FusionInputData, i);
            return fixedShape;
        }).ToArray();


    private Expr Split(Call call, Marker marker, Marker lhsMarker, Marker rhsMarker, SegmentInfo info, int current,
        int limit, Dictionary<Var, int[]> varValues)
    {
        // DumpScope.Current.DumpIR(call, "SplitOriginCall", count.ToString());
        var (inputIndex, dimIndex, segments) = info;
        var newVar = new Var(new TensorType(call.CheckedDataType,
            Enumerable.Repeat(Dimension.Unknown, call.CheckedShape.Rank).ToArray()));
        var sp = ConstantOfShape(ShapeOf(call), Cast(0, call.CheckedDataType));
        int i = 0;
        // todo:造一个合法的require才行
        // var requires = call.Arguments.ToArray()
        // .Select(arg => (Expr)IR.F.Math.Require(false, ConstantOfShape(ShapeOf(arg), Cast(0, arg.CheckedDataType)), "input dim large than limit")).ToArray();
        // var error = new IR.Tuple(requires);
        var body = segments.OrderByDescending(x => x).Aggregate(
            // todo: sp is invalid??
            (Expr)IR.F.Math.Require(false, sp, "input dim large than limit"),
            // (Expr)error,
            (sum, seg) =>
            {
                // Console.WriteLine("segment value");
                // Console.WriteLine(seg);
                var cond = Cast(ShapeOf(call.Arguments[inputIndex]), DataTypes.Int32)[dimIndex] <= seg;
                // select var value for current segment
                var varInfo = varValues.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value[i]));
                var thenBody = current + 1 < limit
                    ? Split(call, marker, lhsMarker, rhsMarker, info, current + 1, limit, varValues)
                    : MakeSplitEntry(call, marker, lhsMarker, rhsMarker, info, seg, InputInfo, varInfo, FusionInputData,
                        FusionInputsShape);
                var elseBody = sum;
                i++;
                return new If(cond, thenBody, elseBody);
            });
        return body;
    }

    public static Expr MakeSplitEntry(Call call, Marker callMaker, Marker lhsMarker, Marker rhsMarker, SegmentInfo info,
        int seg, Dictionary<Var, Expr[]> inputInfo,
        Dictionary<Var, IValue> varInfo, Dictionary<Var, Expr[]> fusionInputdata,
        Dictionary<Var, Expr[]> fusionInputsShape)
    {
        // Console.WriteLine("MakeEntry");
        var fixedShapeList = call.Arguments.ToArray()
            .Select((arg, i) => PreProcess(arg, info, seg, inputInfo, varInfo, fusionInputdata)).ToArray();
        // return new IR.Tuple(fixedShapeList);
        var markers = new[] { lhsMarker, rhsMarker };

        var args = fixedShapeList.Select((arg, i) => markers[i].With(target: arg)).ToArray();
        var then = call.With(arguments: args);
        var thenWithMarker = callMaker.With(target: then);
        thenWithMarker.InferenceType();
        // // Console.WriteLine($"PostShape: {DumpUtility.SerializeShape(thenWithMarker.CheckedShape.ToValueArray())}");
        var post = PostProcess(thenWithMarker, call, fusionInputsShape);
        post.InferenceType();
        // DumpScope.Current.DumpIR(post, "PostExpr");
        return post;
    }

    public static Expr PreProcess(Expr input, SegmentInfo info, int seg, Dictionary<Var, Expr[]> inputInfo,
        Dictionary<Var, IValue> varValues, Dictionary<Var, Expr[]> fusionInputData)
    {
        // Console.WriteLine($"input type {input.GetType().Name}");
        // compute FixedShape by new var value
        // Console.WriteLine("PreProcess");

        var fixedShape = ShapeEvaluate(input, inputInfo, varValues, fusionInputData);
        // return new Call(new BucketPad(), input, fixedShape);

        // Console.WriteLine("fixedShape");
        // Console.WriteLine(string.Join(",", fixedShape));
        // return new Call(new BucketPad(), input, fixedShape);


        var pads = fixedShape - Cast(ShapeOf(input), DataTypes.Int32);
        var paddings = Transpose(Stack(new IR.Tuple(Enumerable.Repeat(0, fixedShape.Length).ToArray(), pads), 0),
            new[] { 1, 0 });
        var fixedInput = IR.F.NN.Pad(input, paddings, PadMode.Constant, Cast(0, input.CheckedDataType));
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
                    // DumpScope.Current.DumpIR(shapeExpr, "mkInputShapeExpr", $"{count}");
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
        // shape.InferenceType();
        DumpScope.Current.DumpIR(shape, "PostShapeExpr", count.ToString());
        var rank = call.CheckedShape.Rank;
        // return Slice(call, Enumerable.Repeat(0, rank).ToArray(), Cast(ShapeOf(call), DataTypes.Int32), rank);

        // // todo: fix this
        return Slice(call, Enumerable.Repeat(0, rank).ToArray(), Cast(shape, DataTypes.Int32), rank);
        //
        return Slice(call, Enumerable.Repeat(0, rank).ToArray(), Cast(ShapeOf(originCall), DataTypes.Int32), rank);


        // 原始的input，在这个op里面根据call的类型，推导出对应的原始的shape
        // 但是这个只针对单个op的情况，如果合并以后怎么办

        //  conv
        //   |
        // matmal
        // matmal关于var的shape表达式？？
        return new Call(new BucketSlice(), call, originCall);
    }
}
//
// [RuleGenerator]
// public partial class MergeChildCallToFusion : IRewriteRule
// {
//     // fusion
//     //   |
//     //  call
//
//     public Expr GetReplace(Call fusionCall, Call call, Fusion fusion)
//     {
//         var body = call;
//         var index = call.Arguments.IndexOf(fusionCall);
//         var fusionBody = fusion.Body;
//         var newBody = ReplaceUtility.ReplaceCallParams(body, (index, fusionBody));
//         var newParams = new[] { };
//         // 更新effect var？
//         return new VarFusion(fusion.Name, fusion.ModuleKind, new[] { }, newBody);
//     }
// }
//
// [RuleGenerator]
// public partial class MergeParentCallToFusion : IRewriteRule
// {
//     //  call
//     //   |
//     // fusion
//     public Expr GetReplace(Call fusionCall, Call call, Fusion fusion)
//     {
//         // call input into fusion
//         var body = (Call)fusion.Body;
//         var index = fusionCall.Arguments.IndexOf(call);
//         var newBody = ReplaceUtility.ReplaceCall(body, (index, call));
//         // input of call is new input
//         return new VarFusion(fusion.Name, fusion.ModuleKind, new[] { }, newBody);
//     }
// }

[RuleGenerator]
public partial class FusionBucket : IRewriteRule
{
    private Dictionary<Var, Expr[]> InputInfo;
    private Dictionary<string, (int, int)> Dict;
    private static int counter = 0;
    private int segmentsCount;

    public FusionBucket(Dictionary<Var, Expr[]> inputInfo, Dictionary<string, (int, int)> dict, int segmentsCount)
    {
        InputInfo = inputInfo;
        Dict = dict;
        this.segmentsCount = segmentsCount;
    }

    public IPattern Pattern => IsCall("call",
        IsFusion("fusion", "stackvm", IsWildcard(), GenerateParameters(null)),
        GenerateParameters(null));


    public Expr? GetReplace(Call call, VarFusion fusion)
    {
        // if (counter > 5)
        // {
        //     return null;
        // }
        Console.WriteLine($"FusionBucketGetReplace {counter}");
        // Console.WriteLine(fusion.Body.Metadata.OutputNames);
        // todo: replace every call in Fusion
        var relPath = $"ShapeBucketDebug/{counter}";
        var ctx = new RunPassContext();
        // ctx.RewriteOnce = true;
        // var sp = call.Arguments.ToArray().Select(x => x.Metadata.ShapeExpr).ToArray();

        // DumpScope.Current.DumpIR(call, "origin", relPath);

        var fusionInputInfo = MakeFusionInputShapeInfo(call, fusion);

        foreach (var value in fusionInputInfo.Values)
        {
            foreach (var expr in value)
            {
                if (!expr.IsAlive)
                {
                    // Console.WriteLine("not alive in fusion bucket");
                    throw new NotImplementedException();
                }
            }
        }

        // Console.WriteLine("Check all live");
        using var _ = new ExprPinner(fusionInputInfo.Values.SelectMany(x => x).ToArray());

        var oldBody = fusion.Body;
        // todo: input info contains fusion var

        foreach (var callArgument in call.Arguments)
        {
            if (callArgument is not Marker)
            {
                throw new InvalidOperationException("not marker in FusionBucket");
            }
        }

        // todo: input info contains fusion var
        var fusionInputShapes = fusion.Parameters
            .ToArray()
            .Zip(call.Arguments.ToArray())
            .ToDictionary(pair => pair.First, pair =>
            {
                var shape = (Expr)ShapeOf(pair.Second);
                return Enumerable.Range(0, pair.Second.CheckedShape.Rank).Select(i => shape[i]).ToArray();
            });
        // DumpScope.Current.DumpIR(fusion.Body, "oldBody", relPath);
        // PrintMinMaxShape("oldBody", call, oldBody, relPath, fusionInputInfo);
        var newBody = CompilerServices.Rewrite(fusion.Body,
            new[]
            {
                new ReplaceRewrite(fusion.EffectVar, InputInfo, fusionInputInfo, Dict, call.Arguments.ToArray(),
                    fusionInputShapes, segmentsCount)
            }, ctx);
        if (oldBody == newBody)
        {
            // Console.WriteLine("no change");
            return null;
        }

        // DumpScope.Current.DumpIR(newBody, "newBody", relPath);
        // PrintMinMaxShape("newBody", call, newBody, relPath, fusionInputInfo);

        if (newBody is Call c && c.Arguments[0] is Call argCall && argCall.Target is FixShape)
        {
            // Console.WriteLine("remove Fusion");
        }

        int n = 0;
        foreach (var callArgument in call.Arguments)
        {
            // DumpScope.Current.DumpIR(callArgument, $"{n++}_arg", relPath);
        }

        // Console.WriteLine("Fusion Bucket ok");
        var args = call.Arguments.ToArray();
        if (newBody is If @if)
        {
            // var letBody = GetItem(IR.F.Math.Require(true, new IR.Tuple(new[]{}.Append(newBody).ToArray())), args.Length - 1);
            // newBody = GetItem(IR.F.Math.Require(true, new IR.Tuple(new[]{@if.Then, @if.Else}.Append(newBody).ToArray())), args.Length - 1);
            // newBody = @if.With(paramList: call.Arguments.ToArray().Append(call));
            newBody = @if.With(paramList: call.Arguments.ToArray());
            for (int i = 0; i < call.Arguments.Length; i++)
            {
                // DumpScope.Current.DumpIR(call.Arguments[i], i.ToString(), "if_args");
            }

            // DumpScope.Current.DumpIR(newBody, $"after_let", relPath);
        }

        // PrintMinMaxShape("newBodyWithLet", call, newBody, relPath, fusionInputInfo);

        // todo: 清除var的range_of
        var body = fusion.Parameters.ToArray().Zip(args).Aggregate(newBody, (sum, pair) =>
        {
            var (param, arg) = pair;
            return ReplaceExpr(sum, param, arg);
        });

        // PrintMinMaxShape("newBodyWithReplace", call, body, relPath, fusionInputInfo);

        // DumpScope.Current.DumpIR(body, $"after", relPath);


        int j = 0;
        foreach (var callArgument in call.Arguments)
        {
            // DumpScope.Current.DumpIR(callArgument, $"{j++}_arg_after_replace", relPath);
        }
        // todo: move marker
        // 1. 在replace里面将var替换为input，在preprocess的时候将marker移动过去
        // 2. 输出的marker在出去的时候自动会添加，但是后面还有一个slice，必须把marker移动过去
        // todo: slice没有range的话那是不是可以再写一个rule进行移动


        // // DumpScope.Current.DumpIR(fusion.Body, "old");
        // var newBody = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray()).Aggregate(oldBody, (sum, pair) =>
        // {
        //     var (param, arg) = pair;
        //     return ReplaceExpr(sum, param, arg);
        // });
        // // DumpScope.Current.DumpIR(fusion.Body, "new");
        //
        // var body = CompilerServices.Rewrite(newBody,
        //     new[] { new ReplaceRewrite(fusion.EffectVar, InputInfo, fusionInputInfo, Dict, call.Arguments.ToArray()) }, ctx);
        // if (body == oldBody)
        // {
        // return null;
        // }

        // unpack fusion
        // Console.WriteLine("after rewrite");
        // PrintMinMaxShape("InEndOfBucket", call, body, relPath, fusionInputInfo);
        DumpScope.Current.DumpIR(body, "BucketResult", relPath);
        // // Console.WriteLine($"bodyShapeExpr {DumpUtility.SerializeShape(minShape)} | {DumpUtility.SerializeShape(maxShape)}");
        counter++;
        return body;
    }

    private void PrintMinMaxShape(string prefix, Call call, Expr body, string relPath,
        Dictionary<Var, Expr[]> fusionInputInfo)
    {
        // Console.WriteLine(prefix);
        var (lhsMin, lhsMax) = ReplaceRewrite.GetBoundDict(InputInfo, Dict);
        var lhsMinShape = ReplaceRewrite.ShapeEvaluate(call.Arguments[0], InputInfo, lhsMin, fusionInputInfo);
        var lhsMaxShape = ReplaceRewrite.ShapeEvaluate(call.Arguments[0], InputInfo, lhsMax, fusionInputInfo);

        var (rhsMin, rhsMax) = ReplaceRewrite.GetBoundDict(InputInfo, Dict);
        var rhsMinShape = ReplaceRewrite.ShapeEvaluate(call.Arguments[1], InputInfo, rhsMin, fusionInputInfo);
        var rhsMaxShape = ReplaceRewrite.ShapeEvaluate(call.Arguments[1], InputInfo, rhsMax, fusionInputInfo);


        var (min, max) = ReplaceRewrite.GetBoundDict(InputInfo, Dict);
        var minShape = ReplaceRewrite.ShapeEvaluate(body, InputInfo, min, fusionInputInfo);
        // Console.WriteLine("Start BodyShapeExpr Max");
        var maxShape = ReplaceRewrite.ShapeEvaluate(body, InputInfo, max, fusionInputInfo);
        // don't have fusion input info
        // var bodyShapeExpr = body.EvaluateShapeExpr(InputInfo);
        // // DumpScope.Current.DumpIR(bodyShapeExpr, "bodyShapeExpr", relPath);


        // var minShape = shapeExpr.Evaluate(min).AsTensor().ToArray<int>();
        // var maxShape = shapeExpr.Evaluate(max).AsTensor().ToArray<int>();
        Console.WriteLine(
            $"{prefix}_{relPath} lhs: {DumpUtility.SerializeShape(lhsMinShape)} | {DumpUtility.SerializeShape(lhsMaxShape)} rhs: {DumpUtility.SerializeShape(rhsMinShape)} | {DumpUtility.SerializeShape(rhsMaxShape)}  bodyShapeExpr {DumpUtility.SerializeShape(minShape)} | {DumpUtility.SerializeShape(maxShape)}");
    }

    private Dictionary<Var, Expr[]> MakeFusionInputShapeInfo(Call call, VarFusion fusion)
    {
        var data = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray().Select((arg, i) =>
        {
            // Console.WriteLine("arg shape eval");
            var result = arg.EvaluateShapeExpr(InputInfo);
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

    public static Expr ReplaceExpr(Expr body, Expr target, Expr expr)
    {
        var mutator = new Passes.Mutators.Substitutor(e =>
        {
            if (ReferenceEquals(e, target))
            {
                return expr;
            }

            return null;
        });
        return mutator.Visit(body, Unit.Default);
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
