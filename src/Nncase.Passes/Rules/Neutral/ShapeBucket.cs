using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using DryIoc.ImTools;
using Nncase.Diagnostics;
using Nncase.IR;
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
        // if (counter > 15)
        // {
        //     return null;
        // }
        //CompilerServices.DumpIR(arg0, "arg0", "/Users/homura/Code/nncase-fix/tests_output/ShapeBucketTest/TestModel/ShapeExpr/");
        // //CompilerServices.DumpIR(arg1, "arg1", "/Users/homura/Code/nncase-fix/tests_output/ShapeBucketTest/TestModel/ShapeExpr/");
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
    public ReplaceRewrite(Var[] var, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, Expr[]> fusionInputData)
    {
        DimVar = var;
        InputInfo = inputInfo;
        FusionInputData = fusionInputData;
    }

    private Dictionary<Var, Expr[]> InputInfo;
    private Dictionary<Var, Expr[]> FusionInputData;
    public IPattern Pattern => IsCall("call", IsWildcard(), GenerateParameters(null));
    private Var[] DimVar;
    private static Call currentcall;

    // compute segment
    // segments count
    public Expr? GetReplace(Call call)
    {
        currentcall = call;
        var dict = new Dictionary<string, (int, int)>
        {
            { "batch", (2, 2) }, { "tok_len", (3, 12) }, { "enc_len", (6, 24) }, { "dec_len", (2, 8) },
        };
        var (minDict, maxDict) = GetBoundDict(InputInfo, dict);

        // 1. compute dim range
        var minFixedShapeList = ComputeFixedShape(call, minDict);
        var maxFixedShapeList = ComputeFixedShape(call, maxDict);


        var counts = minFixedShapeList.Zip(maxFixedShapeList).Select(pair =>
        {
            var (min, max) = pair;
            // minShape maxShape
            return min.Zip(max).Count(pair => pair.First != pair.Second);
        }).ToArray();
        Console.WriteLine("min MatmulShape:" +
                          string.Join("|", minFixedShapeList.Select(s => DumpUtility.SerializeShape(s))) +
                          $" {call.Metadata.OutputNames}" + " max MatmulShape:" +
                          string.Join("|", maxFixedShapeList.Select(s => DumpUtility.SerializeShape(s))) +
                          $" {call.Metadata.OutputNames}. count:{string.Join(",", counts)} var:{DimVar.Count()}");

        // Console.WriteLine("maxShape");
        // Console.WriteLine(string.Join("\n", maxFixedShapeList.Select(s => DumpUtility.SerializeShape(s))));
        // 2. compute segments point

        // batch不变的, 当成2处理

        var inputindex = call.Arguments.ToArray().IndexOf(expr => !expr.CheckedShape.IsFixed);
        var dimIndex = call.Arguments[inputindex].CheckedShape.ToArray().IndexOf(dim => dim.IsUnknown);
        // var expr = Split(call, inputindex, dimIndex, 0, 1, segments);
        // //CompilerServices.DumpIR(expr, "", "/Users/homura/Code/nncase-fix/tests_output/");
        // return expr;
        return call;
    }

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


    private Expr Split(Call call, int inputIndex, int dimIndex, int current, int limit, int[] segments)
    {
        var newVar = new Var(new TensorType(call.CheckedDataType,
            Enumerable.Repeat(Dimension.Unknown, call.CheckedShape.Rank).ToArray()));
        var body = segments.OrderByDescending(x => x).Aggregate(
            (Expr)IR.F.Math.Require(false, newVar, "input dim large than limit"),
            (sum, seg) =>
            {
                // input[i] < seg
                var cond = Cast(ShapeOf(call.Arguments[inputIndex]), DataTypes.Int32)[dimIndex] <= seg;
                var varInfo = new Dictionary<Var, IValue> { { DimVar[0], Value.FromTensor(seg) } };
                var thenBody = current + 1 < limit
                    ? Split(call, inputIndex, dimIndex, current + 1, limit, segments)
                    : MakeSplitEntry(call, InputInfo, varInfo);
                var elseBody = sum;
                return new If(cond, thenBody, elseBody);
            });
        return body;
    }

    public static Expr MakeSplitEntry(Call call, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, IValue> varInfo)
    {
        var fixedShapeList = call.Arguments.ToArray().Select(arg => PreProcess(arg, inputInfo, varInfo)).ToArray();
        var then = call.With(arguments: fixedShapeList);
        return PostProcess(then);
    }

    public static Expr PreProcess(Expr input, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, IValue> varInfo)
    {
        return input;
        // var fixedShape = ShapeEvaluate(input, inputInfo, varInfo);
        // var pads = fixedShape - Cast(ShapeOf(input), DataTypes.Int32);
        // var paddings = Transpose(Stack(new IR.Tuple(Enumerable.Repeat(0, fixedShape.Length).ToArray(), pads), 0),
        //     new[] { 1, 0 });
        // var fixedInput = IR.F.NN.Pad(input, paddings, PadMode.Constant, Cast(0, input.CheckedDataType));
        // // todo:
        // // 1. reshape的问题
        // // 2. 消除pad和slice的问题，这个不应该去消除，而是应该在要分段的时候判断插入分段的位置，不然很难消除
        // return fixedInput;
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
                foreach (var expr1 in pair.Value)
                {
                    Console.WriteLine(expr1);
                    Console.WriteLine(expr1.IsAlive);
                    Console.WriteLine(expr1.CheckedDataType);
                }

                Console.WriteLine("end");
                // 包含了input的shape表达式，必须想办法区分开才行
                var shape = Stack(new IR.Tuple(pair.Value.Select(x => Cast(x, DataTypes.Int32)).ToArray()), 0).Evaluate(varInfo).AsTensor();
                Console.WriteLine(pair.Key.Name);
                Console.WriteLine(string.Join(",", pair.Value.Select(x => x.ToString()).ToArray()));
                return ConstantOfShape(
                    shape,
                    Cast(0, pair.Key.CheckedDataType)).Evaluate();
            });
        Console.WriteLine("dummyInput");
        var newEvaluatorInfo = dummyInput.Concat(varInfo).ToDictionary(pair => pair.Key, pair => pair.Value);
        Console.WriteLine("new info");
        var shapeExpr =
            expr.EvaluateShapeExpr(info.Concat(fusionInfo).ToDictionary(pair => pair.Key, pair => pair.Value));
        shapeExpr.InferenceType();
        //CompilerServices.DumpIR(shapeExpr, "ShapeExpr",
            // "/Users/homura/Code/nncase-fix/tests_output/ShapeBucketTest/TestModel/ShapeExpr");
        Console.WriteLine("evaluate shapeExpr");
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

    public FusionBucket(Dictionary<Var, Expr[]> inputInfo)
    {
        InputInfo = inputInfo;
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

        //CompilerServices.DumpIR(call, "origin",
            // "/Users/homura/Code/nncase-fix/tests_output/ShapeBucketTest/TestModel/ShapeExpr");

        int i = 0;
        var data = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray().Select(arg =>
        {
            var result = arg.EvaluateShapeExpr(InputInfo);
            Console.WriteLine("Before Infer");
            result.InferenceType();
            Console.WriteLine("Infer ok");
            //CompilerServices.DumpIR(result, $"result{i++}",
                // "/Users/homura/Code/nncase-fix/tests_output/ShapeBucketTest/TestModel/ShapeExpr");
            return Enumerable.Range(0, arg.CheckedShape.Rank).Select(i => result[i]).ToArray();
        })).Select(pair => new KeyValuePair<Var, Expr[]>(pair.First, pair.Second));
        var fusionInputData = data.ToDictionary(pair => pair.Key, pair => pair.Value);

        // todo: input info contains fusion var
        var newBody = CompilerServices.Rewrite(fusion.Body,
            new[] { new ReplaceRewrite(fusion.EffectVar, InputInfo, fusionInputData) }, ctx);
        // unpack fusion
        Console.WriteLine("after rewrite");
        var body = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray()).Aggregate(newBody, (sum, pair) =>
        {
            var (param, arg) = pair;
            return ReplaceExpr(sum, param, arg);
        });
        return body;
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
