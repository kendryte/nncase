// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.Utility;

[assembly: InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Passes.Rules;

/// <summary>
/// auto distributed the xpu fusion.
/// </summary>
[RuleGenerator]
public sealed partial class AutoPacking : IRewriteRule
{
    public IPattern Pattern { get; } = IsCallWildcard("call", IsFusion("fusion", CPUTarget.Kind, IsWildcard("body"), IsVArgsRepeat("parameters", () => IsVar())));

    private Expr? GetReplace(Call call, Fusion fusion, Expr body, IReadOnlyList<Expr> parameters, IReadOnlyList<Expr> callParams)
    {
        // 1. convert to distribute graph
        if (body is Call { Target: Unpack } || (body is IR.Tuple tp && tp.Fields.AsValueEnumerable().Any(e => e is Call { Target: Unpack })))
        {
            return null;
        }

        var distConverter = new AutoPackingConvertVisitor();
        var newbody = distConverter.Convert(body);
        var newFusion = fusion.With(moduleKind: CPUTarget.Kind, body: newbody, parameters: parameters.Cast<Var>().ToArray());
        return new Call(newFusion, callParams.ToArray());
    }
}

internal sealed class AutoPackingConvertVisitor : ExprVisitor<List<Expr>, Unit>
{
    public Expr Convert(Expr body)
    {
        var equivalents = Visit(body);
        var graph = new EGraph();
        foreach (var (exprKey, candidates) in ExprMemo.Where(kv => kv.Key is not Op))
        {
            Unions(graph, candidates);
        }

        var root = Unions(graph, equivalents);

        // run pass
        var post = CompilerServices.ERewrite(
            graph,
            new IRewriteRule[] {
                new Passes.Rules.Neutral.FoldConstCall(),
                new Passes.Rules.CPU.FoldPackUnpack(),
                new Passes.Rules.CPU.FoldPackConcatUnpack(),
            },
            new());
        return post.Extract(root, null, out _);
    }

    protected override List<Expr> DefaultVisitLeaf(Expr expr)
    {
        return new() { expr };
    }

    protected override List<Expr> VisitLeafCall(Call expr)
    {
        if (expr.Target is not Op op)
        {
            throw new NotSupportedException("not support auto distributed call function");
        }

        var candidates = op switch
        {
            IR.NN.Softmax => new Passes.Rules.CPU.PackSoftmax().GetReplace(expr),
            IR.NN.Swish => new Passes.Rules.CPU.PackSwish().GetReplace(expr),
            IR.NN.LayerNorm => new Passes.Rules.CPU.PackLayerNorm().GetReplace(expr),
            IR.Math.MatMul => new Passes.Rules.CPU.PackMatMul().GetReplace(expr),
            IR.Math.Unary => new Passes.Rules.CPU.PackUnary().GetReplace(expr),
            IR.Math.Binary => new Passes.Rules.CPU.PackBinary().GetReplace(expr),
            Transpose => new Passes.Rules.CPU.PackTranspose().GetReplace(expr),
            Unsqueeze => new Passes.Rules.CPU.PackUnsqueeze().GetReplace(expr),
            Reshape => new Passes.Rules.CPU.PackReshape().GetReplace(expr),
            Slice => new Passes.Rules.CPU.PackSlice().GetReplace(expr),
            _ => new() { },
        };

        var ret = new List<Expr> { expr };
        ret.AddRange(candidates);
        return ret;
    }

    private EClass Unions(EGraph graph, IEnumerable<Expr> equivalents)
    {
        var eids = equivalents.Select(graph.Add).ToArray();
        foreach (var cls in eids.Skip(1))
        {
            graph.Union(eids[0], cls);
        }

        graph.Rebuild();
        return eids[0];
    }
}
