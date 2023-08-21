// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using DryIoc.ImTools;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.ShapeBucket;

public static class CallValidator
{
    private static readonly HashSet<RuntimeTypeHandle> ForceConvert = new()
    {
        typeof(Conv2D).TypeHandle,
        typeof(MatMul).TypeHandle,
        typeof(Unsqueeze).TypeHandle,
        typeof(Squeeze).TypeHandle,
        typeof(Cast).TypeHandle,
        typeof(Unary).TypeHandle,
        typeof(Transpose).TypeHandle,
        typeof(Pad).TypeHandle,
    };
    // todo: add debug mode

    private static readonly HashSet<RuntimeTypeHandle> MaybeDynamic = new()
    {
        // typeof(SpaceToBatch).TypeHandle,
        // typeof(BatchToSpace).TypeHandle,
        typeof(Concat).TypeHandle,
        typeof(Stack).TypeHandle,
        typeof(Binary).TypeHandle,
        typeof(Slice).TypeHandle,
        typeof(Gather).TypeHandle,
        typeof(ShapeOf).TypeHandle,
        // typeof(Reshape).TypeHandle,
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

    public static bool IsForceConvert(Expr target) => ForceConvert.Contains(target.GetType().TypeHandle);

    public static bool ValidTarget(Expr target)
    {
        if (target is ActivationOp)
        {
            return true;
        }

        if (IsMaybeDynamic(target) || IsForceConvert(target))
        {
            return true;
        }

        return false;
    }
}

public static class ShapeBucketHelper
{
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
            // DumpScope.Current.DumpIR(expr, s, reletivePath);
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

internal class OpCounter : ExprVisitor<Expr, Unit>
{
    private Dictionary<RuntimeTypeHandle, int> counter;

    protected override Expr VisitCall(Call expr)
    {
        if (expr.Target is Op)
        {
            var handle = expr.Target.GetType().TypeHandle;
            if (counter.ContainsKey(handle))
            {
                counter[handle] += 1;
            }
            else
            {
                counter[handle] = 1;
            }
        }

        return base.VisitCall(expr);
    }
}
