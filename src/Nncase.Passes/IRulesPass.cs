// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;

namespace Nncase.Passes;

/// <summary>
/// Rules addable.
/// </summary>
public interface IRulesAddable
{
    /// <summary>
    /// Add the rewrite rule.
    /// </summary>
    /// <typeparam name="T">Descriptor type.</typeparam>
    /// <param name="parameters">Descriptor's constructor parameters.</param>
    /// <returns>Add result.</returns>
    AddResult<T> Add<T>(params object[] parameters)
        where T : class, IRewriteRule;

    /// <summary>
    /// Add the rewrite rule.
    /// </summary>
    /// <param name="ruleType">Descriptor type.</param>
    /// <param name="parameters">Descriptor's constructor parameters.</param>
    /// <returns>Add result.</returns>
    AddResult<IRewriteRule> Add(Type ruleType, params object[] parameters);

    /// <summary>
    /// Add rule result.
    /// </summary>
    /// <typeparam name="T">Descriptor type.</typeparam>
    public struct AddResult<T> : IRulesAddable
        where T : class, IRewriteRule
    {
        private readonly IRulesPass _rulesPass;

        internal AddResult(IRulesPass rulesPass, T rule)
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

/// <summary>
/// Pass contains rewrite rules.
/// </summary>
public interface IRulesPass : IPass, IRulesAddable
{
    /// <summary>
    /// Gets rules.
    /// </summary>
    IReadOnlyList<IRewriteRule> Rules { get; }
}
