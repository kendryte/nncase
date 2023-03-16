// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;

namespace Nncase.Passes;

public class EGraphRulesPass : EGraphPass, IRulesPass
{
    private readonly List<IRewriteRule> _rules = new();
    private readonly IEGraphRewriteProvider _rewriteProvider;

    /// <summary>
    /// Initializes a new instance of the <see cref="EGraphRulesPass"/> class.
    /// </summary>
    public EGraphRulesPass()
    {
        _rewriteProvider = CompileSession.GetRequiredService<IEGraphRewriteProvider>();
    }

    /// <inheritdoc/>
    public IReadOnlyList<IRewriteRule> Rules => _rules;

    /// <inheritdoc/>
    public IRulesAddable.AddResult<T> Add<T>(params object[] parameters)
        where T : class, IRewriteRule
    {
        var compileSession = ((IPassIntern)this).CompileSession;
        using var scope = new CompileSessionScope(compileSession);
        var rule = ActivatorUtilities.CreateInstance<T>(compileSession, parameters);
        _rules.Add(rule);
        return new(this, rule);
    }

    /// <inheritdoc/>
    public IRulesAddable.AddResult<IRewriteRule> Add(Type ruleType, params object[] parameters)
    {
        var compileSession = ((IPassIntern)this).CompileSession;
        using var scope = new CompileSessionScope(compileSession);
        var rule = (IRewriteRule)ActivatorUtilities.CreateInstance(compileSession, ruleType, parameters);
        _rules.Add(rule);
        return new(this, rule);
    }

    /// <inheritdoc/>
    protected override Task<IEGraph> RunCoreAsync(IEGraph eGraph, RunPassContext context)
    {
        _rewriteProvider.ERewrite(eGraph, Rules, context);
        return Task.FromResult(eGraph);
    }
}
