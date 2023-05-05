using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using DryIoc.ImTools;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
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

    public VarFusion(string moduleKind, Expr body, ReadOnlySpan<Var> parameters, Var[] effectVar) : base(moduleKind, body,
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
    public IPattern Pattern => IsMatMul(
        IsWildcard("arg0"),
        IsWildcard("arg1"));

    public MatmulToFusion(Dictionary<Var, Expr[]> varMap)
    {
        VarMap = varMap;
    }

    private Dictionary<Var, Expr[]> VarMap;

    public Expr? GetReplace(Expr arg0, Expr arg1)
    {
        counter++;
        // if (counter > 3)
        // {
        //     return null;
        // }
        DumpScope.Current.DumpIR(arg0, "arg0");
        DumpScope.Current.DumpIR(arg1, "arg1");
        var visitor = new FindVar();
        Console.WriteLine("argExpr0");
        var arg0Expr = arg0.EvaluateShapeExpr(VarMap);
        Console.WriteLine("argExpr1");
        var arg1Expr = arg1.EvaluateShapeExpr(VarMap);
        Console.WriteLine("visit");
        visitor.Visit(arg0Expr);
        visitor.Visit(arg1Expr);
        var set = visitor.Vars.ToHashSet().Except(VarMap.Keys.ToHashSet()).ToHashSet();
        var lhs = new Var(arg0.CheckedType);
        var rhs = new Var(arg1.CheckedType);
        var m = IR.F.Math.MatMul(lhs, rhs);
        // Debug.Assert(set.Count <= 1);
        Console.WriteLine("GetReplace Set");
        foreach (var var in set)
        {
            Console.WriteLine(var.Name);
        }

        Console.WriteLine("GetReplace End");

        var f = new VarFusion("matmul", "stackvm", set.ToArray(), m, lhs, rhs);
        var c = new Call(f, arg0, arg1);
        return c;
    }
}

[RuleGenerator]
public partial class ReplaceRewrite : IRewriteRule
{
    public ReplaceRewrite(Var[] var, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, Expr[]> fusionInputData, Dictionary<string, (int, int)> dict)
    {
        DimVar = var;
        InputInfo = inputInfo;
        FusionInputData = fusionInputData;
        this.dict = dict;
    }

    private Dictionary<Var, Expr[]> InputInfo;
    private Dictionary<Var, Expr[]> FusionInputData;
    private Dictionary<string, (int, int)> dict;
    public IPattern Pattern => IsCall("call", IsWildcard(), GenerateParameters(null));
    private Var[] DimVar;
    private static Call currentcall;

    public record SegmentInfo(int InputIndex, int DimIndex, int[] Segments);

    // compute segment
    // segments count
    public Expr? GetReplace(Call call)
    {
        var segmentCount = 2;
        currentcall = call;
        var (minDict, maxDict) = GetBoundDict(InputInfo, dict);

        Console.WriteLine("ComputeFixedShape");
        // 1. compute dim range
        var minFixedShapeList = ComputeFixedShape(call, minDict);
        var maxFixedShapeList = ComputeFixedShape(call, maxDict);

        Console.WriteLine("GetDimInfo");
        // 2. get dim info(inputIndex, (dimIndex, range)
        var counts = minFixedShapeList.Zip(maxFixedShapeList).Select((pair, inputIndex) =>
        {
            Console.WriteLine(inputIndex);
            var (min, max) = pair;
            // (range, dimIndex)
            var range = Enumerable.Range(0, min.Length).Zip(min.Zip(max)).Where(data =>
            {
                var (dimIndex, pair) = data;
                return pair.First != pair.Second;
            }).ToArray();
            return (inputIndex, range);
        }).Where(pair => pair.range.Length > 0).ToArray();
        var totalCount = counts.Length;

        PrintMatmulInfo(call, minFixedShapeList, maxFixedShapeList, totalCount);
        PrintSegmentInfo(counts);
        Console.WriteLine($"total count {totalCount}");
        if (minFixedShapeList[0].SequenceEqual(maxFixedShapeList[0]) &&
            minFixedShapeList[1].SequenceEqual(maxFixedShapeList[1]))
        {
            // todo: 做了不该做的，写个test验证下
            Console.WriteLine("same, not be process");
            return null;
        }
        // todo: no effect, fix this
        if (totalCount == 0)
        {
            Console.WriteLine("total count = 0, not be process");
            return null;
        }
        var varValues = MakeVarValuesForAllSegment(segmentCount);
        // 代入所有的segments
        var info = ComputeSegmentInfo(counts, segmentCount);
        var expr = Split(call, info, 0, 1, varValues);
        expr.InferenceType();
        DumpScope.Current.DumpIR(expr, "SplitResult");
        return expr;
    }

    private static SegmentInfo ComputeSegmentInfo(
        (int inputIndex, (int First, (int First, int Second) Second)[] range)[] counts, int segmentCount)
    {
        var (iIndex, dimIndex, (min, max)) = counts.Select(x =>
        {
            Debug.Assert(x.range.Length == 1);
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
            Console.WriteLine("segments list");
            Console.WriteLine($"{min}, {max}");
            Console.WriteLine($"{((max - min) / segmentCount)}");
            Console.WriteLine(string.Join(",", segments));
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


    private Expr Split(Call call, SegmentInfo info, int current, int limit, Dictionary<Var, int[]> varValues)
    {
        var (inputIndex, dimIndex, segments) = info;
        var newVar = new Var(new TensorType(call.CheckedDataType,
            Enumerable.Repeat(Dimension.Unknown, call.CheckedShape.Rank).ToArray()));
        var sp = ConstantOfShape(ShapeOf(call), Cast(0, call.CheckedDataType));
        int i = 0;
        // todo:造一个合法的require才行
        var body = segments.OrderByDescending(x => x).Aggregate(
            // todo: sp is invalid??
            (Expr)IR.F.Math.Require(false, sp, "input dim large than limit"),
            (sum, seg) =>
            {
                Console.WriteLine("segment value");
                Console.WriteLine(seg);
                var cond = Cast(ShapeOf(call.Arguments[inputIndex]), DataTypes.Int32)[dimIndex] <= seg;
                // select var value for current segment
                var varInfo = varValues.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value[i]));
                var thenBody = current + 1 < limit
                    ? Split(call, info, current + 1, limit, varValues)
                    : MakeSplitEntry(call, info, seg, InputInfo, varInfo, FusionInputData);
                var elseBody = sum;
                i++;
                return new If(cond, thenBody, elseBody);
            });
        return body;
    }

    public static Expr MakeSplitEntry(Call call, SegmentInfo info, int seg, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, IValue> varInfo, Dictionary<Var, Expr[]> fusionInputdata)
    {
        Console.WriteLine("MakeEntry");
        var fixedShapeList = call.Arguments.ToArray().Select((arg, i) => PreProcess(arg, info, seg, inputInfo, varInfo, fusionInputdata)).ToArray();
        var then = call.With(arguments: fixedShapeList);
        return PostProcess(then);
    }

    public static Expr PreProcess(Expr input, SegmentInfo info, int seg, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, IValue> varValues, Dictionary<Var, Expr[]> fusionInputData)
    {
        // compute FixedShape by new var value
        Console.WriteLine("PreProcess");
        var fixedShape = ShapeEvaluate(input, inputInfo, varValues, fusionInputData);
        Console.WriteLine("fixedShape");
        Console.WriteLine(string.Join(",", fixedShape));
        var pads = fixedShape - Cast(ShapeOf(input), DataTypes.Int32);
        var paddings = Transpose(Stack(new IR.Tuple(Enumerable.Repeat(0, fixedShape.Length).ToArray(), pads), 0),
        new[] { 1, 0 });
        var fixedInput = IR.F.NN.Pad(input, paddings, PadMode.Constant, Cast(0, input.CheckedDataType));
        return new Call(new FixShape(), fixedInput, fixedShape);
    }

    private static int count = 0;

    public static int[] ShapeEvaluate(Expr expr, Dictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo,
        Dictionary<Var, Expr[]> fusionInfo, int i = 0)
    {
        var i32Info = varInfo.ToDictionary(pair => pair.Key,
            pair => (IValue)Value.FromTensor(pair.Value.AsTensor().ToScalar<int>()));
        // info to fixed shape and return constant of shape
        var dummyInput = info.ToDictionary(pair => pair.Key,
            pair =>
            {
                Console.WriteLine("mkinput");
                // foreach (var expr1 in pair.Value)
                // {
                //     Console.WriteLine(expr1);
                //     Console.WriteLine(expr1.IsAlive);
                //     Console.WriteLine(expr1.CheckedDataType);
                // }
                //
                // Console.WriteLine("end");
                // 包含了input的shape表达式，必须想办法区分开才行
                try
                {
                    var shapeExpr = Stack(new IR.Tuple(pair.Value.Select(x => Cast(x, DataTypes.Int32)).ToArray()), 0);
                    DumpScope.Current.DumpIR(shapeExpr, "mkInputShapeExpr");
                    var shape = shapeExpr.Evaluate(varInfo).AsTensor();
                    // Console.WriteLine(pair.Key.Name);
                    // Console.WriteLine(string.Join(",", pair.Value.Select(x => x.ToString()).ToArray()));
                    return ConstantOfShape(
                        shape,
                        Cast(0, pair.Key.CheckedDataType)).Evaluate();
                }
                catch (Exception e)
                {
                    Console.WriteLine(e);
                    throw;
                }
            });
        Console.WriteLine("dummyInput");
        var newEvaluatorInfo = dummyInput.Concat(varInfo).ToDictionary(pair => pair.Key, pair => pair.Value);
        Console.WriteLine("new info");
        var shapeExpr =
            expr.EvaluateShapeExpr(info.Concat(fusionInfo).ToDictionary(pair => pair.Key, pair => pair.Value));
        shapeExpr.InferenceType();
        DumpScope.Current.DumpIR(shapeExpr, "ShapeExpr");
        Console.WriteLine("evaluate shapeExpr");
        // todo:this error
        // todo: shape expr inference
        var shape = shapeExpr.Evaluate(newEvaluatorInfo);
        Console.WriteLine("after evaluate shapeExpr");

        return shape.AsTensor().ToArray<int>();
    }

    public static Call PostProcess(Call call)
    {
        var rank = call.CheckedShape.Rank;
        return Slice(call, Enumerable.Repeat(0, rank).ToArray(), Cast(ShapeOf(call), DataTypes.Int32), rank);
    }
}

[RuleGenerator]
public partial class FusionBucket : IRewriteRule
{
    private Dictionary<Var, Expr[]> InputInfo;
    private Dictionary<string, (int, int)> Dict;
    public FusionBucket(Dictionary<Var, Expr[]> inputInfo, Dictionary<string, (int, int)> dict)
    {
        InputInfo = inputInfo;
        Dict = dict;
    }

    public IPattern Pattern => IsCall("call",
        IsFusion("fusion", "stackvm", IsWildcard(), GenerateParameters(null)),
        GenerateParameters(null));


    public Expr? GetReplace(Call call, VarFusion fusion)
    {
        Console.WriteLine("FusionBucketGetReplace");
        // todo: replace every call in Fusion
        var ctx = new RunPassContext();
        // ctx.RewriteOnce = true;
        // var sp = call.Arguments.ToArray().Select(x => x.Metadata.ShapeExpr).ToArray();

        DumpScope.Current.DumpIR(call, "origin");

        // todo: 这个地方报错
        var fusionInputInfo = MakeFusionInputShapeInfo(call, fusion);

        var oldBody = fusion.Body;
        // todo: input info contains fusion var
        var newBody = CompilerServices.Rewrite(fusion.Body,
            new[] { new ReplaceRewrite(fusion.EffectVar, InputInfo, fusionInputInfo, Dict) }, ctx);
        if (oldBody == newBody)
        {
            return null;
        }
        // unpack fusion
        Console.WriteLine("after rewrite");
        var body = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray()).Aggregate(newBody, (sum, pair) =>
        {
            var (param, arg) = pair;
            return ReplaceExpr(sum, param, arg);
        });
        return body;
    }

    private Dictionary<Var, Expr[]> MakeFusionInputShapeInfo(Call call, VarFusion fusion)
    {
        var data = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray().Select((arg, i) =>
        {
            Console.WriteLine("arg shape eval");
            var result = arg.EvaluateShapeExpr(InputInfo);
            Console.WriteLine("Before Infer");
            result.InferenceType();
            Console.WriteLine("Infer ok");
            DumpScope.Current.DumpIR(result, $"result{i++}");
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
