// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using DryIoc.ImTools;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

public class VarFusion : Fusion
{
    public Var EffectVar;

    public VarFusion(string name, string moduleKind, Expr body, ReadOnlySpan<Var> parameters, Var effectVar)
        : base(name, moduleKind, body, parameters)
    {
        EffectVar = effectVar;
    }

    public VarFusion(string moduleKind, Expr body, ReadOnlySpan<Var> parameters, Var effectVar)
        : base(moduleKind, body, parameters)
    {
        EffectVar = effectVar;
    }

    public VarFusion(string name, string moduleKind, Var effectVar, Expr body, params Var[] parameters)
        : base(name, moduleKind, body, parameters)
    {
        EffectVar = effectVar;
    }

    public VarFusion(string moduleKind, Var effectVar, Expr body, params Var[] parameters)
        : base(moduleKind, body, parameters)
    {
        EffectVar = effectVar;
    }
}

[RuleGenerator]
public partial class MatmulToFusion : IRewriteRule
{
    private readonly Dictionary<Var, Expr[]> _varMap;

    public MatmulToFusion(Dictionary<Var, Expr[]> varMap)
    {
        _varMap = varMap;
    }

    public IPattern Pattern => IsMatMul(
        IsWildcard("arg0"),
        IsWildcard("arg1"));

    public Expr? GetReplace(Expr arg0, Expr arg1)
    {
        var visitor = new FindVar();
        visitor.Visit(arg0.EvaluateShapeExpr(_varMap));
        visitor.Visit(arg1.EvaluateShapeExpr(_varMap));
        var set = visitor.Vars.ToHashSet().Except(_varMap.Keys.ToHashSet()).ToHashSet();
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

        var f = new VarFusion("matmul", "stackvm", set.Count == 0 ? null : set.ToArray()[0], m, lhs, rhs);
        var c = new Call(f, arg0, arg1);
        return c;
    }
}

[RuleGenerator]
public partial class ReplaceRewrite : IRewriteRule
{
    private static Call _currentcall;

    private readonly Dictionary<Var, Expr[]> _inputInfo;

    private readonly Dictionary<Var, Expr[]> _fusionInputData;

    public ReplaceRewrite(Var var, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, Expr[]> fusionInputData)
    {
        _dimVar = var;
        _inputInfo = inputInfo;
        _fusionInputData = fusionInputData;
    }

    private readonly Var _dimVar;

    public IPattern Pattern => IsCall("call", IsWildcard(), GenerateParameters(null));

    public static (Dictionary<Var, IValue> MinDict, Dictionary<Var, IValue> MaxDict) GetBoundDict(Dictionary<Var, Expr[]> inputInfo, Dictionary<string, (int Min, int Max)> limitDict)
    {
        // find vars in Input ShapeExpr
        var vars = inputInfo.Values.SelectMany(x => x).OfType<Var>().ToHashSet().ToArray();

        // DimVarName -> Dict.key -> Dict.Value
        var minDict = limitDict.ToDictionary(
            pair => vars.FindFirst(v => v.Name == pair.Key),
            pair =>
            {
                return (IValue)Value.FromTensor((long)pair.Value.Min);
            });
        var maxDict = limitDict.ToDictionary(
            pair => vars.FindFirst(v => v.Name == pair.Key),
            pair => (IValue)Value.FromTensor((long)pair.Value.Max));
        return (minDict, maxDict);
    }

    public static Expr MakeSplitEntry(Call call, Dictionary<Var, Expr[]> inputInfo, Dictionary<Var, IValue> varInfo)
    {
        var fixedShapeList = call.Arguments.ToArray().Select(arg => PreProcess(arg, inputInfo, varInfo)).ToArray();
        var then = call.With(arguments: fixedShapeList);
        return PostProcess(then);
    }

    // compute segment
    // segments count
    public Expr? GetReplace(Call call)
    {
        _currentcall = call;
        var dict = new Dictionary<string, (int, int)>
        {
            { "batch", (24, 48) }, { "tok_len", (24, 48) }, { "enc_len", (24, 48) }, { "dec_len", (24, 48) },
        };
        var (minDict, maxDict) = GetBoundDict(_inputInfo, dict);

        // 1. compute dim range
        var minFixedShapeList = ComputeFixedShape(call, minDict);
        var maxFixedShapeList = ComputeFixedShape(call, maxDict);

        Console.WriteLine("minShape");
        Console.WriteLine(string.Join("\n", minFixedShapeList.Select(s => DumpUtility.SerializeShape(s).ToArray())));
        Console.WriteLine("maxShape");
        Console.WriteLine(string.Join("\n", maxFixedShapeList.Select(s => DumpUtility.SerializeShape(s).ToArray())));

        // 2. compute segments point

        // batch不变的, 当成2处理
        var inputindex = call.Arguments.ToArray().IndexOf(expr => !expr.CheckedShape.IsFixed);
        var dimIndex = call.Arguments[inputindex].CheckedShape.ToArray().IndexOf(dim => dim.IsUnknown);

        // var expr = Split(call, inputindex, dimIndex, 0, 1, segments);
        // CompilerServices.DumpIR(expr, "", "/Users/homura/Code/nncase-fix/tests_output/");
        // return expr;
        return call;
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

    private int[][] ComputeFixedShape(Call call, Dictionary<Var, IValue> varInfo) =>
        call.Arguments.ToArray().Select(arg =>
        {
            var fixedShape = ShapeEvaluate(arg, _inputInfo, varInfo, _fusionInputData);
            return fixedShape;
        }).ToArray();

    private Expr Split(Call call, int inputIndex, int dimIndex, int current, int limit, int[] segments)
    {
        var newVar = new Var(new TensorType(call.CheckedDataType, Enumerable.Repeat(Dimension.Unknown, call.CheckedShape.Rank).ToArray()));
        var body = segments.OrderByDescending(x => x).Aggregate(
            (Expr)IR.F.Math.Require(false, newVar, "input dim large than limit"),
            (sum, seg) =>
            {
                // input[i] < seg
                var cond = Cast(ShapeOf(call.Arguments[inputIndex]), DataTypes.Int32)[dimIndex] <= seg;
                var varInfo = new Dictionary<Var, IValue> { { _dimVar, Value.FromTensor((long)seg) } };
                var thenBody = current + 1 < limit
                    ? Split(call, inputIndex, dimIndex, current + 1, limit, segments)
                    : MakeSplitEntry(call, _inputInfo, varInfo);
                var elseBody = sum;
                return new If(cond, thenBody, elseBody);
            });
        return body;
    }

    public static int[] ShapeEvaluate(Expr expr, Dictionary<Var, Expr[]> info, Dictionary<Var, IValue> varInfo, Dictionary<Var, Expr[]> fusionInfo)
    {
        var i32Info = varInfo.ToDictionary(pair => pair.Key, pair => (IValue)Value.FromTensor(pair.Value.AsTensor().ToScalar<int>()));

        // info to fixed shape and return constant of shape
        var dummyInput = info.ToDictionary(
            pair => pair.Key,
            pair =>
            {
                // 包含了input的shape表达式，必须想办法区分开才行
                var shape = Stack(new IR.Tuple(pair.Value), 0).Evaluate(varInfo).AsTensor();
                Console.WriteLine(pair.Key.Name);
                Console.WriteLine(string.Join(",", pair.Value.Select(x => x.ToString()).ToArray()));
                return ConstantOfShape(
                    shape,
                    Cast(0, pair.Key.CheckedDataType)).Evaluate();
            });
        Console.WriteLine("dummyInput");
        var newEvaluatorInfo = dummyInput.Concat(varInfo).ToDictionary(pair => pair.Key, pair => pair.Value);
        Console.WriteLine("new info");
        var shapeExpr = expr.EvaluateShapeExpr(info.Concat(fusionInfo).ToDictionary(pair => pair.Key, pair => pair.Value));
        CompilerServices.DumpIR(shapeExpr, "ShapeExpr", "/Users/homura/Code/nncase-fix/tests_output/ShapeBucketTest/TestModel/ShapeExpr");
        Console.WriteLine("evaluate shapeExpr");
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
    private readonly Dictionary<Var, Expr[]> _inputInfo;

    public FusionBucket(Dictionary<Var, Expr[]> inputInfo)
    {
        _inputInfo = inputInfo;
    }

    public IPattern Pattern => IsCall(
        "call",
        IsFusion("fusion", "stackvm", IsWildcard(), GenerateParameters(null)),
        GenerateParameters(null));

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

    public Expr? GetReplace(Call call, VarFusion fusion)
    {
        Console.WriteLine("FusionBucketGetReplace");

        // todo: replace every call in Fusion
        var ctx = new RunPassContext();

        // ctx.RewriteOnce = true;
        // var sp = call.Arguments.ToArray().Select(x => x.Metadata.ShapeExpr).ToArray();
        int i = 0;
        var data = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray().Select(arg =>
        {
            var result = arg.EvaluateShapeExpr(_inputInfo);
            CompilerServices.DumpIR(result, $"result{i++}", "/Users/homura/Code/nncase-fix/tests_output/ShapeBucketTest/TestModel/ShapeExpr");
            return Enumerable.Range(0, arg.CheckedShape.Rank).Select(i => result[i]).ToArray();
        })).Select(pair => new KeyValuePair<Var, Expr[]>(pair.First, pair.Second));
        var fusionInputData = data.ToDictionary(pair => pair.Key, pair => pair.Value);

        CompilerServices.DumpIR(call, "origin", "/Users/homura/Code/nncase-fix/tests_output/ShapeBucketTest/TestModel/ShapeExpr");

        // todo: input info contains fusion var
        var newBody = CompilerServices.Rewrite(fusion.Body, new[] { new ReplaceRewrite(fusion.EffectVar, _inputInfo, fusionInputData) }, ctx);

        // unpack fusion
        Console.WriteLine("after rewrite");
        var body = fusion.Parameters.ToArray().Zip(call.Arguments.ToArray()).Aggregate(newBody, (sum, pair) =>
        {
            var (param, arg) = pair;
            return ReplaceExpr(sum, param, arg);
        });
        return body;
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
