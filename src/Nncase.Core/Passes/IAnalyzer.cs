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
    IReadOnlyCollection<Type> RequiredAnalysisResultTypes { get; }

    void Invalidate(Expr key);

    void InvalidateAll();

    void OnBeginIRMutate(IReadOnlyDictionary<Expr, Expr> mutateMemo);
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
