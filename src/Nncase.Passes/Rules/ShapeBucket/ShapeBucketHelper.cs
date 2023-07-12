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

namespace Nncase.Passes.Rules.ShapeBucket;

internal static class ShapeBucketHelper
{
    public static void PrintEffectVar(string name, Var[] set)
    {
        Console.WriteLine($"{name} EffectVar:");
        foreach (var var in set)
        {
            Console.WriteLine(var.Name);
        }
    }

    // avoid dup marker user
    public static T DupExpr<T>(T body)
        where T : Expr
    {
        T dupFusionBody = body switch
        {
            Marker m => (T)(object)m.With(target: DupExpr(m.Target)),
            Call c => (T)(object)c.With(),
            IR.Tuple t => (T)(object)new IR.Tuple(t.Fields.ToArray().Select(DupExpr).ToArray()),
            _ => body,
        };
        return dupFusionBody;
    }

    public static Var[] MakeEffectVarArray(Dictionary<Var, Expr[]> varMap, params Expr[] args)
    {
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

    internal static void DumpIR(Expr expr, string prefix, string? reletivePath = null, string? printPrefix = null)
    {
        // if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
        {
            Console.WriteLine($"{printPrefix} {prefix}");
            DumpScope.Current.DumpIR(expr, prefix, reletivePath);
        }
    }
}

internal static class ExprArrayExtension
{
    public static IEnumerable<Expr> OfNoConst(this IEnumerable<Expr> args)
    {
        return args.Where(x => x is not TensorConst);
    }
}

public class FindExpr : ExprVisitor<Expr, Unit>
{
    private Func<Expr, bool> f;
    private List<Expr> list = new();
    private Expr[] limit = { };

    public List<Expr> Run(Expr expr, Expr[] limit, Func<Expr, bool> checker)
    {
        f = checker;
        this.limit = limit;
        Visit(expr);
        return list;
    }

    protected override Expr DefaultVisitLeaf(Expr expr)
    {
        if (f(expr))
        {
            list.Add(expr);
        }

        return expr;
    }

    protected override Expr DispatchVisit(Expr expr)
    {
        if (limit.Contains(expr))
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

public static class CallValidator
{
    static readonly Dictionary<RuntimeTypeHandle, int> OpList = new()
    {
        // { typeof(Reshape).TypeHandle, 0 },
        { typeof(Unsqueeze).TypeHandle, 0 },
        { typeof(Squeeze).TypeHandle, 0 },
        // btm
        // { typeof(Slice).TypeHandle, 0 },
        { typeof(Concat).TypeHandle, 0 },
        { typeof(Cast).TypeHandle, 0 },
        { typeof(Stack).TypeHandle, 0 },
        { typeof(Expand).TypeHandle, 0 },
        // { typeof(ConstantOfShape).TypeHandle, 0 },
        { typeof(Where).TypeHandle, 0 },
        { typeof(Compare).TypeHandle, 0 },
        { typeof(Gather).TypeHandle, 0 },

        // compute
        // maybe Reduce.Prod only, for eval shape
        { typeof(Reduce).TypeHandle, 1 },
        { typeof(Transpose).TypeHandle, 1 },
        // marker
        { typeof(Unary).TypeHandle, 1 },
        { typeof(Binary).TypeHandle, 2 },
        { typeof(Clamp).TypeHandle, 2 },
        { typeof(Pad).TypeHandle, 2 },

        // ...
        { typeof(Conv2D).TypeHandle, 2 },
        { typeof(MatMul).TypeHandle, 2 },
        { typeof(Tile).TypeHandle, 0 },
    };

    public static bool ValidTarget(Expr target)
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
