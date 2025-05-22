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
using Microsoft.Extensions.Primitives;
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

    public async Task<IReadOnlyDictionary<IVar, IValue>> GetFirstElementAsync(IAsyncEnumerable<IReadOnlyDictionary<IVar, IValue>> asyncEnumerable)
    {
        await foreach (var element in asyncEnumerable)
        {
            return element;
        }

        return null!;
    }

    public float CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
    {
        if (vectorA.Length != vectorB.Length)
        {
            Console.WriteLine("Array lengths are inconsistent and cosine similarity cannot be calculated!");
            return 0;
        }

        float dotProduct = 0;
        float magnitudeA = 0;
        float magnitudeB = 0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            magnitudeA += vectorA[i] * vectorA[i];
            magnitudeB += vectorB[i] * vectorB[i];
        }

        magnitudeA = (float)Math.Sqrt(magnitudeA);
        magnitudeB = (float)Math.Sqrt(magnitudeB);

        if (magnitudeA == 0 || magnitudeB == 0)
        {
            return 0; // 避免除以零
        }

        return dotProduct / (magnitudeA * magnitudeB);
    }

    public async Task UpdateQuantConfigBySensitivity(BaseFunction function, RunPassContext options)
    {
        string dumpDir = Directory.GetCurrentDirectory() + "/tests_output/test_debug";
        string cpuResultDir = dumpDir + "/cpu_result_0.txt";
        CompilerServices.DumpIR(((Nncase.IR.Function)function).Body, "matmul", dumpDir);
        var samples = CompileSession.CompileOptions.QuantizeOptions.CalibrationDataset!.Samples;
        var first = await GetFirstElementAsync(samples);
        var result = CompilerServices.Evaluate(((Nncase.IR.Function)function).Body, first).AsTensor().ToArray<float>();

        float[] expected = File.ReadAllLines(cpuResultDir)
                                  .Skip(1)
                                  .Select(line => float.Parse(line.Trim()))
                                  .ToArray();

        float cosineSimilarity = CalculateCosineSimilarity(expected, result);
        Console.ForegroundColor = ConsoleColor.Blue;
        Console.WriteLine("Please Check the IR in:" + dumpDir);
        Console.WriteLine($"Evaluate Cosine Similarity: {cosineSimilarity}");
        Console.ResetColor();
    }

    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction function, RunPassContext options)
    {
        _ = UpdateQuantConfigBySensitivity(function, options);
        return Task.FromResult((BaseFunction)CompilerServices.Rewrite(function, Rules, options));
    }
}
