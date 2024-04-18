// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

public sealed class CPUFunctionPartitionPass : ModulePass
{
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext context)
    {
        var funcs = module.Functions.Count;
        for (int i = 0; i < funcs; i++)
        {
            if (module.Functions[i] is Function function)
            {
                Function pre = function;
                Function post;

                // int count = 0;
                while (true)
                {
                    var rewriter = new PartitionRewriter();
                    post = (Function)rewriter.Rewrite(pre, default);
                    if (!rewriter.IsMutated)
                    {
                        break;
                    }

                    // CompilerServices.DumpDotIR(post, count++.ToString(), "/Users/lisa/Documents/nncase/tests_output");
                    pre = post;
                }

                module.Replace(i, post);
            }
        }

        return Task.FromResult(module);
    }
}

internal sealed class PartitionContext
{
    private readonly HashSet<Expr> _entryPoints = new(ReferenceEqualityComparer.Instance);

    public void AddEntryPoint(Expr expr)
    {
        _entryPoints.Add(expr);
    }

    public Dictionary<Expr, Expr> CreateVarMaps() => _entryPoints.Select(e => (e, new Var(e.CheckedType))).ToDictionary(kv => kv.e, kv => (Expr)kv.Item2, (IEqualityComparer<Expr>)ReferenceEqualityComparer.Instance);
}

internal sealed class PartitionVisitor : ExprVisitor<Unit, Unit, PartitionContext>
{
    protected override Unit VisitLeafCall(Call expr, PartitionContext context)
    {
        if (expr is Call { Target: IR.CPU.Boxing { NewType: DistributedType } })
        {
            context.AddEntryPoint(expr.Arguments[0]);
        }

        return default;
    }

    protected override Unit DefaultVisitLeaf(Expr expr, PartitionContext context) => default;
}

internal sealed class PartitionRewriter : ExprRewriter<Unit>
{
    protected override Expr RewriteLeafCall(Call expr, Unit context)
    {
        if (!IsMutated && expr is Call { Target: IR.CPU.Boxing { NewType: TensorType } })
        {
            var visitor = new PartitionVisitor();
            var ctx = new PartitionContext();
            visitor.Visit(expr, ctx);
            var mps = ctx.CreateVarMaps();

            var cloner = new ReplacingExprCloner(mps);
            var post = cloner.Clone(expr, default);
            var parameters = mps.Values.OfType<Var>().ToArray();
            return new Call(new Function(post, parameters).With(moduleKind: Targets.CPUTarget.Kind), mps.Keys.ToArray());
        }

        return expr;
    }
}
