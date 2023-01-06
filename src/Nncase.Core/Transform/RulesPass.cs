// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.TIR;

namespace Nncase.Transform;

/// <summary>
/// Rules addable.
/// </summary>
public interface IRulesAddable
{
    /// <summary>
    /// Add the rewrite rule.
    /// </summary>
    /// <typeparam name="T">Rule type.</typeparam>
    /// <param name="parameters">Rule's constructor parameters.</param>
    /// <returns>Add result.</returns>
    RulesPass.AddResult<T> Add<T>(params object[] parameters)
        where T : class, IRewriteRule;

    /// <summary>
    /// Add the rewrite rule.
    /// </summary>
    /// <param name="ruleType">Rule type.</param>
    /// <param name="parameters">Rule's constructor parameters.</param>
    /// <returns>Add result.</returns>
    RulesPass.AddResult<IRewriteRule> Add(Type ruleType, params object[] parameters);
}

/// <summary>
/// Pass contains rewrite rules.
/// </summary>
public abstract class RulesPass : FunctionPass, IRulesAddable
{
    private readonly List<IRewriteRule> _rules = new();

    /// <summary>
    /// Gets rules.
    /// </summary>
    public IReadOnlyList<IRewriteRule> Rules => _rules;

    /// <inheritdoc/>
    public AddResult<T> Add<T>(params object[] parameters)
        where T : class, IRewriteRule
    {
        using var scope = new CompileSessionScope(CompileSession);
        var rule = ActivatorUtilities.CreateInstance<T>(CompileSession, parameters);
        _rules.Add(rule);
        return new(this, rule);
    }

    /// <inheritdoc/>
    public AddResult<IRewriteRule> Add(Type ruleType, params object[] parameters)
    {
        using var scope = new CompileSessionScope(CompileSession);
        var rule = (IRewriteRule)ActivatorUtilities.CreateInstance(CompileSession, ruleType, parameters);
        _rules.Add(rule);
        return new(this, rule);
    }

    /// <summary>
    /// Add rule result.
    /// </summary>
    /// <typeparam name="T">Pass type.</typeparam>
    public struct AddResult<T> : IRulesAddable
        where T : class, IRewriteRule
    {
        private readonly RulesPass _rulesPass;

        internal AddResult(RulesPass rulesPass, T rule)
        {
            _rulesPass = rulesPass;
            Rule = rule;
        }

        /// <summary>
        /// Gets rule.
        /// </summary>
        public T Rule { get; }

        /// <inheritdoc/>
        public AddResult<TRule> Add<TRule>(params object[] parameters)
            where TRule : class, IRewriteRule => _rulesPass.Add<TRule>(parameters);

        /// <inheritdoc/>
        public AddResult<IRewriteRule> Add(Type ruleType, params object[] parameters)
            => _rulesPass.Add(ruleType, parameters);

        /// <summary>
        /// Configure rule.
        /// </summary>
        /// <param name="configureRule">Configure rule action.</param>
        /// <returns>This add result.</returns>
        public AddResult<T> Configure(Action<T> configureRule)
        {
            configureRule(Rule);
            return this;
        }
    }
}
