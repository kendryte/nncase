// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Passes;

/// <summary>
/// Dataflow pass.
/// </summary>
public class QuantizerChecker : FunctionPass, IRulesPass
{
    private readonly List<IRewriteRule> _rules = new();
    private readonly List<Type> _analysisTypes = new();

    /// <inheritdoc/>
    public IReadOnlyList<IRewriteRule> Rules => _rules;

    /// <inheritdoc/>
    public override IReadOnlyCollection<Type> AnalysisTypes => _analysisTypes;

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

    public void AddAnalysis<T>()
        where T : IAnalysisResult
    {
        _analysisTypes.Add(typeof(T));
    }

    public async Task<IReadOnlyDictionary<Var, IValue>> GetFirstElementAsync(IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> asyncEnumerable)
    {
        await foreach (var element in asyncEnumerable)
        {
            return element;
        }

        return null!;
    }

    public async Task UpdateQuantConfigBySensitivity(BaseFunction function, RunPassContext options)
    {
        CompilerServices.DumpIR(((Nncase.IR.Function)function).Body, "matmul", "/compiler3.0/dev3.0/nncase/tests_output/test_debug");
        var samples = CompileSession.CompileOptions.QuantizeOptions.CalibrationDataset!.Samples;
        var first = await GetFirstElementAsync(samples);
        var curentResults = CompilerServices.Evaluate(((Nncase.IR.Function)function).Body, first).AsTensor().ToArray<float>();
    }

    /// <inheritdoc/>
    protected override async Task<BaseFunction> RunCoreAsync(BaseFunction function, RunPassContext options)
    {
        await UpdateQuantConfigBySensitivity(function, options);
        return null!;
    }
}
