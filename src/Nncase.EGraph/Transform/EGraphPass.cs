// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

/// <summary>
/// EGraph pass.
/// </summary>
public class EGraphPass : RulesPass
{
    private readonly IEGraphRewriteProvider _rewriteProvider;
    private readonly Evaluator.IBaseFuncCostEvaluator? _baseFuncCostEvaluator;

    /// <summary>
    /// Initializes a new instance of the <see cref="EGraphPass"/> class.
    /// </summary>
    /// <param name="baseFuncCostEvaluator">Extenal cost evaluator.</param>
    public EGraphPass(Evaluator.IBaseFuncCostEvaluator? baseFuncCostEvaluator = null)
    {
        _rewriteProvider = CompileSession.GetRequiredService<IEGraphRewriteProvider>();
        _baseFuncCostEvaluator = baseFuncCostEvaluator;
    }

    /// <inheritdoc/>
    protected override async Task<BaseFunction> RunCoreAsync(BaseFunction function, RunPassContext context)
    {
        // note 这里需要避免egraph处理了多function的导致update失败.
        var c = new FunctionCollector();
        c.Visit(function);
        if (c.Functions.Count > 1)
        {
            return function;
        }

        var graph = new EGraph();
        var root = graph.Add(function);
        _rewriteProvider.ERewrite(graph, Rules, context);
        await OnPostRewriteStartAsync(graph, context);
        await OnPostRewriteAsync(graph, context);
        await OnPostRewriteEndAsync(graph, context);
        var post = graph.Extract(root, _baseFuncCostEvaluator);
        CompilerServices.InferenceType(post);
        return (BaseFunction)post;
    }

    /// <summary>
    /// The callback after egraph rewrite.
    /// </summary>
    /// <param name="eGraph">EGraph after rewrite.</param>
    /// <param name="context">Run pass context.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    protected virtual Task OnPostRewriteAsync(EGraph eGraph, RunPassContext context)
    {
        return Task.CompletedTask;
    }

    /// <summary>
    /// The callback function you can custom process func with run pass options.
    /// </summary>
    /// <param name="eGraph">EGraph after rewrite.</param>
    /// <param name="context">Run pass context.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    protected virtual Task OnPostRewriteStartAsync(EGraph eGraph, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            using var fs = DumpScope.Current.OpenFile(Path.Combine("PostRewriteStart", $"V{eGraph.Version}.dot"));
            EGraphPrinter.DumpEgraphAsDot(eGraph, null, fs);
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// The callback function you can custom process func with run pass options.
    /// </summary>
    /// <param name="eGraph">EGraph after post rewrite.</param>
    /// <param name="context">Run pass context.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    protected virtual Task OnPostRewriteEndAsync(EGraph eGraph, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            using var fs = DumpScope.Current.OpenFile(Path.Combine("PostRewriteEnd", $"V{eGraph.Version}.dot"));
            EGraphPrinter.DumpEgraphAsDot(eGraph, null, fs);
        }

        return Task.CompletedTask;
    }

    public class FunctionCollector : ExprVisitor<int, IRType>
    {
        public HashSet<Function> Functions = new(ReferenceEqualityComparer.Instance);

        public override int VisitLeaf(Function expr)
        {
            Functions.Add(expr);
            return 0;
        }

        public override int DefaultVisitLeaf(Expr expr) => 1;
    }
}
