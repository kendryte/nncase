// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using DryIoc.ImTools;
using Nncase.CodeGen;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeExpr;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.ShapeBucket;

public static class CallValidator
{
    public static readonly HashSet<RuntimeTypeHandle> ForceConvert = new()
    {
        typeof(Conv2D).TypeHandle,
        typeof(Conv2DTranspose).TypeHandle,
        typeof(MatMul).TypeHandle,
        typeof(Transpose).TypeHandle,
        typeof(Pad).TypeHandle,
        typeof(Tile).TypeHandle,
    };

    private static readonly HashSet<RuntimeTypeHandle> CauseDynamic = new()
    {
        typeof(Reshape).TypeHandle, typeof(IR.Tensors.Range).TypeHandle
    };

    private static readonly HashSet<RuntimeTypeHandle> ComputeCanBeMerge = new()
    {
        typeof(Unary).TypeHandle, typeof(Tile).TypeHandle, typeof(Binary).TypeHandle,
    };

    // todo: add debug mode
    private static readonly HashSet<RuntimeTypeHandle> MaybeDynamic = new()
    {
        typeof(Concat).TypeHandle,
        typeof(Stack).TypeHandle,
        typeof(Binary).TypeHandle,
        typeof(Slice).TypeHandle,
        typeof(Gather).TypeHandle,
        typeof(ShapeOf).TypeHandle,

        typeof(Unsqueeze).TypeHandle,
        typeof(Squeeze).TypeHandle,
        typeof(Cast).TypeHandle,
        typeof(Unary).TypeHandle,

        typeof(Reshape).TypeHandle,
        typeof(Expand).TypeHandle,
        typeof(ConstantOfShape).TypeHandle,
        typeof(Where).TypeHandle,
        typeof(Compare).TypeHandle,
        typeof(Reduce).TypeHandle,
        typeof(Clamp).TypeHandle,
        typeof(Tile).TypeHandle,
        typeof(CumSum).TypeHandle,
        typeof(IR.Tensors.Range).TypeHandle,
    };

    public static bool IsMaybeDynamic(Expr target) => MaybeDynamic.Contains(target.GetType().TypeHandle);

    public static bool IsForceConvert(Expr target) => ForceConvert.Contains(target.GetType().TypeHandle) || target is ActivationOp;

    public static bool ValidTarget(Call call, bool greedy)
    {
        var target = call.Target;

        var singleVar = true;
        if (IsForceConvert(target))
        {
            return true;
        }

        // dynamic reshape cause dynamic shape call
        if (!greedy && IsDynamicReshape(call))
        {
            return false;
        }

        if (greedy && IsMaybeDynamic(target))
        {
            return true;
        }

        return false;
    }

    private static bool IsDynamicReshape(Call call) => call.Target is Reshape && call.Arguments[Reshape.Shape.Index] is not Const;
}

public static class ShapeBucketRegister
{
    public static void CheckShapeBucketOptions(ShapeBucketOptions options)
    {
        if (options.Enable)
        {
            if (options.SegmentsCount < 2)
            {
                throw new InvalidOperationException("SegmentsCount should >= 2");
            }
        }
    }

    public static void MergeOp(IPassManager iPassManager, bool greedy)
    {
        iPassManager.AddWithName<DataflowPass>("MergeNextCall").Configure(c =>
        {
            c.Add<MergeNextCallToFusion>(greedy);
            c.Add<MergeNextMarkerToFusion>();
        });
        iPassManager.AddWithName<DataflowPass>("MergePrevCall").Configure(c =>
        {
            c.Add<MergePrevCallToFusion>(greedy);
            c.Add<MergePrevMarkerToFusion>();
        });
    }

    public static void ToFusion(IPassManager p, bool onlyDynamic = false) =>
        p.AddWithName<DataflowPass>("ToFusion").Configure(c =>
        {
            c.Add<FoldRepeatMarker>();
            c.Add<MatmulToFusion>(onlyDynamic);
            c.Add<Conv2DToFusion>(onlyDynamic);
            c.Add<TFConv2DTransposeToFusion>(onlyDynamic);
            c.Add<Conv2DTransposeToFusion>(onlyDynamic);
        });

    public static void Bucket(IPassManager p)
    {
        var shapeList = new Dictionary<BucketFusion, FusionShapeData[]>();
        p.Add<RecordFusionShape>(shapeList);
        p.AddWithName<DataflowPass>("FusionBucket").Configure(c =>
        {
            c.Add<FusionBucket>(shapeList);
        });
    }

    public static void Rebuild(IPassManager p, bool singleVar)
    {
        // rebuild
        ToFusion(p, true);
        MergeOp(p, false);
        // todo: lost to fusion
        p.AddWithName<DataflowPass>("LostToFusion").Configure(p =>
        {
            p.Add<TransposeToFusion>(true);
            p.Add<ActToFusion>(true);
            p.Add<PadToFusion>(true);
        });

        MergeFusion(p, singleVar, false);
        Bucket(p);
    }

    public static void MergeFusion(IPassManager p, bool singleVar, bool greedy)
    {
        if (!singleVar)
        {
            return;
        }

        p.AddWithName<MergeBucketFusionPass>("MergeBucketFusionPass", greedy);
    }

    public static void LostToFusion(IPassManager p, bool singleVar) =>
        p.AddWithName<DataflowPass>("LostToFusion").Configure(c =>
        {
            c.Add<TransposeToFusion>();
            c.Add<UnaryToFusion>();
            c.Add<ActToFusion>();
            c.Add<PadToFusion>();
            if (singleVar)
            {
                c.Add<BinaryToFusion>();
            }
        });

    public static void ClearMarker(IPassManager p) =>
        p.AddWithName<DataflowPass>("ClearSomeMarker").Configure(p =>
        {
            p.Add<ClearFusionOuterMarker>();
            p.Add<RemoveMarker>();
        });

    public static void Simplify(IPassManager p) =>
        p.AddWithName<DataflowPass>("Simplify").Configure(c =>
        {
            c.Add<FoldRepeatMarker>();
            c.Add<FoldStackGetItem>();
            c.Add<FoldConstCall>();
            c.Add<FoldShapeOf>();
            c.Add<FoldTwoReshapes>();
            c.Add<FoldTwoCasts>();
            c.Add<FoldTwoSlices>();
            c.Add<FoldNopBinary>();
            c.Add<FoldNopCast>();
            c.Add<FoldNopReshape>();
            c.Add<FoldNopSlice>();
            c.Add<FoldIf>();
            c.Add<FoldSplitShapeOf>();
            c.Add<FoldBroadcastShape>();
            c.Add<FoldBroadcastShapeConst>();
        });
}

public static class ShapeBucketHelper
{
    public static Dictionary<T, IValue> ConcatDictionary<T>(Dictionary<T, IValue> memo, Dictionary<T, IValue> exprValues)
        where T: Expr
    {
        foreach (var (key, value) in exprValues)
        {
            memo[key] = value;
        }

        return memo;
    }

    public static Dictionary<Var, int[]> MakeVarValuesForAllSegment(ShapeBucketOptions options)
    {
        int segmentCount = options.SegmentsCount;
        var varRange = options.RangeInfo;
        var varMap = options.VarMap;
        var staticShape = false;
        var varAndInputAllSegment = varRange.ToDictionary(pair => pair.Key, pair =>
        {
            var (min, max) = pair.Value;
            if (staticShape)
            {
                return Enumerable.Range(min, max - min).ToArray();
            }

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

    public static int[] ComputeSegmentList(int segmentCount, int min, int max)
    {
        var size = (max - min) / segmentCount;
        return Enumerable.Range(0, segmentCount - 1).Select(i => min + (i * size)).Append(max).ToArray();
    }

    public static void ArgsChecker(Expr[] newArgs)
    {
        if (newArgs.Length == 0)
        {
            throw new InvalidOperationException("Empty Arg");
        }

        if (newArgs.Any(arg => arg is Var v && v.Name.StartsWith("var_")))
        {
            throw new InvalidOperationException("Args has Var in fusion");
        }

        if (newArgs.Any(arg => arg is Marker m && m.Target is Const))
        {
            throw new InvalidOperationException("Args has tuple");
        }

        if (newArgs.Any(arg => arg is IR.Tuple))
        {
            throw new InvalidOperationException("Args has tuple");
        }

        if (newArgs.ToHashSet().Count != newArgs.Length)
        {
            throw new InvalidOperationException("Has Repeat args");
        }
    }

    // clone origin Expr and Do replace for var
    public static Expr ReplaceClone(Expr originBody, params (Var, Expr)[] originVarAndExpr)
    {
        var call = originBody.Clone();
        var finder = new FindVar();
        finder.Visit(call);
        var newVars = finder.Vars;
        originVarAndExpr.ForEach(pair =>
        {
            var (v, newExpr) = pair;
            var varShouldBeReplaced = newVars.FindFirst(newVar => newVar.Name == v.Name);
            if (varShouldBeReplaced == null)
            {
                throw new InvalidOperationException();
            }

            ReplaceExpr(call, varShouldBeReplaced, newExpr);
        });
        return call;
    }

    public static void PrintEffectVar(string name, Var[] set)
    {
        Console.WriteLine($"{name} EffectVar:");
        foreach (var var in set)
        {
            Console.WriteLine(var.Name);
        }
    }

    public static Var[] InputDimVars(CompileSession session)
    {
        return session.CompileOptions.ShapeBucketOptions.VarMap.Values.SelectMany(x => x).OfType<Var>()
            .ToHashSet().ToArray();
    }

    public static Var[] MakeEffectVarArray(CompileSession session, Dictionary<Var, Expr[]> varMap, params Expr[] args)
    {
        var dimVars = InputDimVars(session);
        if (dimVars.Length == 1)
        {
            return dimVars;
        }

        if (dimVars.Length == 0)
        {
            // todo: process this, in test should not have this
            // throw new InvalidOperationException("MaybeError");
        }

        var visitor = new FindVar();
        args.ForEach(arg =>
        {
            DumpIR(arg, "argExpr");
            var argShapeExpr = arg.EvaluateShapeExpr(varMap);
            visitor.Visit(argShapeExpr);
        });
        var vars = visitor.Vars.ToHashSet();

        // PrintEffectVar("VisitorVars", vars.ToArray());
        var inputAndDimVarMap =
            varMap.ToDictionary(pair => pair.Key, pair => pair.Value.OfType<Var>().ToHashSet().ToArray());
        var allDimVars = varMap.Values.SelectMany(x => x).OfType<Var>();
        var afterProcessVars = vars.SelectMany(var =>
        {
            if (inputAndDimVarMap.TryGetValue(var, out var dimVars))
            {
                return dimVars;
            }

            if (allDimVars.Contains(var))
            {
                return new[] { var };
            }

            return new[] { var };
        }).ToHashSet();
        return afterProcessVars.Intersect(allDimVars).ToHashSet().ToArray();
    }

    public static void DumpIR(Expr expr, string prefix, string? reletivePath = null, string? printPrefix = null)
    {
        // if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
        {
            var s = prefix;
            if (prefix.Length > 80)
            {
                s = s[..80];
            }

            Console.WriteLine($"{printPrefix} {prefix}");
            DumpScope.Current.DumpIR(expr, s, reletivePath);
        }
    }

    public static void CheckRepeat(Expr call)
    {
        // todo: 检查所有fusion里面的param有没有重复名字的
        // todo: 检查有没有fusion名字重复的
        var c = new CheckFusionCallVisitor();
        c.Visit(call);
        c.Check();
    }

    public static void CheckErrorVar(Expr body, Var[] vars)
    {
        var f = new FindVar();
        f.Visit(body);
        if (!f.Vars.All(vars.Contains))
        {
            Console.WriteLine(string.Join(", ", f.Vars.Select(x => x.Name).ToArray()));
            throw new InvalidOperationException("Has Invalid Var In Body");
        }
    }

    public static void CheckIRRing(Expr expr)
    {
        var c = new CheckRing();
        c.Visit(expr);
        if (c.ErrList.Count != 0)
        {
            throw new InvalidOperationException("IR has ring");
        }
    }
}

public class FindExpr : ExprVisitor<Expr, Unit>
{
    private readonly List<Expr> _list = new();
    private Func<Expr, bool>? _f;
    private Expr[] _limit = Array.Empty<Expr>();
    private Expr? _outerCall;

    public List<Expr> Run(Expr expr, Expr[] limit, Expr outerCall, Func<Expr, bool> checker)
    {
        _f = checker;
        _outerCall = outerCall;
        _limit = limit;
        Visit(expr);
        return _list;
    }

    protected override Expr DefaultVisitLeaf(Expr expr)
    {
        if (_f!(expr))
        {
            _list.Add(expr);
        }

        return expr;
    }

    protected override Expr DispatchVisit(Expr expr)
    {
        if (_limit.Contains(expr))
        {
            _list.Add(expr);
            return expr;
        }

        if (expr == _outerCall)
        {
            return expr;
        }

        if (HasVisited(expr, out var result))
        {
            return result;
        }

        return MarkVisited(expr, base.DispatchVisit(expr));
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

[RuleGenerator]
public sealed partial class ForceConvertOpChecker : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsCall(
        "call",
        IsOp<Op>(op => CallValidator.IsForceConvert(op)),
        GenerateParameters(null, IsWildcard()));

    // todo: is slice
    public Expr? GetReplace(Call call)
    {
        if (!call.CheckedShape.IsFixed)
        {
            throw new InvalidOperationException("ForceConvertCall should has fixed shape after bucket");
        }

        return call;
    }
}

internal static class ExprArrayExtension
{
    public static IEnumerable<Expr> OfNoConst(this IEnumerable<Expr> args)
    {
        return args.Where(x => x is not TensorConst);
    }
}

internal class KeyValuePairKeyComparer : IEqualityComparer<KeyValuePair<Expr, Var[]>>
{
    public bool Equals(KeyValuePair<Expr, Var[]> x, KeyValuePair<Expr, Var[]> y)
    {
        return Equals(x.Key, y.Key);
    }

    public int GetHashCode(KeyValuePair<Expr, Var[]> obj)
    {
        return HashCode.Combine(obj.Key);
    }
}

public class OpCounter : ExprVisitor<Expr, Unit>
{
    public readonly Dictionary<RuntimeTypeHandle, int> _counter = new();
    public readonly HashSet<Op> OpSet = new();

    protected override Expr VisitCall(Call expr)
    {
        if (expr.Target is Op op)
        {
            var handle = expr.Target.GetType().TypeHandle;
            if (_counter.ContainsKey(handle))
            {
                _counter[handle] += 1;
            }
            else
            {
                _counter[handle] = 1;
                // todo: op能去重吗
                OpSet.Add(op);
            }
        }

        return base.VisitCall(expr);
    }

    protected override Expr DefaultVisitLeaf(Expr expr) => expr;
}

public class CheckRing : ExprVisitor<Expr, Unit>
{
    public List<Expr> ErrList = new();

    protected override Expr DefaultVisitLeaf(Expr expr)
    {
        if (expr.Users.Any(user => user.Users.Contains(expr)))
        {
            ErrList.Add(expr);
        }

        return expr;
    }
}

internal sealed class CheckFusionCallVisitor : ExprWalker
{
    private readonly HashSet<string> _callName = new();
    private readonly Dictionary<string, (string, BucketFusion)> _errorFusion = new();

    private readonly HashSet<string> _fusionName = new();
    private readonly HashSet<string> _repeatFusion = new();

    private readonly HashSet<string> _fusionParamsName = new();
    private readonly HashSet<string> _repeatParamFusion = new();

    public void Check()
    {
        var error = false;
        if (_errorFusion.Count != 0)
        {
            error = true;
            Console.WriteLine("errorFusion");
        }

        if (_repeatFusion.Count != 0)
        {
            error = true;
            Print("repeatFusion not zero", _repeatFusion);
        }

        if (_repeatParamFusion.Count != 0)
        {
            error = true;
            Print("repeatParamFusion not zero", _repeatParamFusion);
        }

        if (error)
        {
            throw new InvalidOperationException();
        }
    }

    protected override Unit VisitLeafFusion(Fusion fusion)
    {
        // 可能有多个user啊，每次进来访问
        if (fusion is BucketFusion bf)
        {
            if (_fusionName.Contains(bf.Name))
            {
                _repeatFusion.Add(bf.Name);
            }
            else
            {
                _fusionName.Add(bf.Name);
            }

            var parameters = bf.Parameters.ToArray();
            foreach (var parameter in parameters)
            {
                if (_fusionParamsName.Contains(parameter.Name))
                {
                    _repeatParamFusion.Add(parameter.Name);
                }
            }

            _fusionParamsName.UnionWith(parameters.Select(p => p.Name).ToArray());
        }

        return default;
    }

    private void Print(string name, HashSet<string> list)
    {
        Console.WriteLine(name);
        foreach (string s in list)
        {
            Console.WriteLine(s);
        }
    }
}
