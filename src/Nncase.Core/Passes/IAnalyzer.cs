// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

/// <summary>
/// Analyzer.
/// </summary>
public interface IAnalyzer
{
    IAnalysisResult Result { get; }

    /// <summary>
    /// Gets required collection of <see cref="IAnalysisResult"/>.
    /// </summary>
    IReadOnlyCollection<Type> RequiredAnalysisResultTypes => Array.Empty<Type>();

    void Invalidate(Expr key);

    void InvalidateAll();
}

/// <summary>
/// Analysis pass.
/// </summary>
public interface IAnalyzer<out TResult> : IAnalyzer
    where TResult : IAnalysisResult
{
    new TResult Result { get; }

    IAnalysisResult IAnalyzer.Result => Result;
}

public interface IAnalyzerFactory
{
    Type ResultType { get; }

    IAnalyzer Activate(Either<BaseFunction, IEGraph> functionOrEGraph);
}

public interface IAnalyzerFactory<TResult> : IAnalyzerFactory
    where TResult : IAnalysisResult
{
    Type IAnalyzerFactory.ResultType => typeof(TResult);

    new IAnalyzer<TResult> Activate(Either<BaseFunction, IEGraph> functionOrEGraph);

    IAnalyzer IAnalyzerFactory.Activate(Either<BaseFunction, IEGraph> functionOrEGraph) => Activate(functionOrEGraph);
}

public interface IAnalyzerManager
{
    IAnalyzerFactory GetFactory(Type resultType);

    T GetAnaylsis<T>(Either<BaseFunction, IEGraph> functionOrEGraph)
        where T : IAnalysisResult
        => (T)GetFactory(typeof(T)).Activate(functionOrEGraph).Result;
}
