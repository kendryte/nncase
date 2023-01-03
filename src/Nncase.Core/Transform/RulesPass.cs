// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.IR;

namespace Nncase.Transform;

/// <summary>
/// Pass contains rewrite rules.
/// </summary>
public abstract class RulesPass : FunctionPass
{
    private readonly List<IRewriteRule> _rules = new();

    /// <summary>
    /// Gets rules.
    /// </summary>
    public IReadOnlyList<IRewriteRule> Rules => _rules;

    /// <summary>
    /// Add the rewrite rule.
    /// </summary>
    /// <typeparam name="T">Rule type.</typeparam>
    /// <param name="configureRule">Configure rule action.</param>
    /// <param name="parameters">Rule's constructor parameters.</param>
    /// <returns>This rule pass.</returns>
    public RulesPass Add<T>(Action<T>? configureRule, params object[] parameters)
        where T : class, IRewriteRule
    {
        using var scope = new CompileSessionScope(CompileSession);
        var rule = ActivatorUtilities.CreateInstance<T>(CompileSession.ServiceProvider, parameters);
        configureRule?.Invoke(rule);
        _rules.Add(rule);
        return this;
    }
}
