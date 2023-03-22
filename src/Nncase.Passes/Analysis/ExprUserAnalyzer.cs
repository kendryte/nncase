// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes.Analysis;

public interface IExprUserAnalysisResult : IAnalysisResult
{
    /// <summary>
    /// Gets users.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Users.</returns>
    IEnumerable<Expr> this[Expr expr] { get; }
}

internal sealed class ExprUserAnalysisResult : IExprUserAnalysisResult
{
    public IEnumerable<Expr> this[Expr expr]
    {
        get
        {
            // If expr is Var, exclude the use in function's parameters.
            if (expr is Var)
            {
                return expr.Users.Where(x => x is not BaseFunction);
            }

            return expr.Users;
        }
    }
}

internal sealed class ExprUserAnalyzer : IAnalyzer<ExprUserAnalysisResult>
{
    public ExprUserAnalyzer(BaseFunction baseFunction)
    {
    }

    public ExprUserAnalysisResult Result { get; } = new();

    public void Invalidate(Expr key)
    {
    }

    public void InvalidateAll()
    {
    }
}

internal sealed class ExprUserAnalyzerFactory : IAnalyzerFactory<IExprUserAnalysisResult>
{
    public IAnalyzer<IExprUserAnalysisResult> Activate(Either<BaseFunction, IEGraph> functionOrEGraph)
    {
        if (functionOrEGraph.Is<IEGraph>())
        {
            throw new ArgumentException($"{nameof(ExprUserAnalyzer)} does not support egraph.", nameof(functionOrEGraph));
        }

        return new ExprUserAnalyzer(functionOrEGraph.Value1);
    }
}
