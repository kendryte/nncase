// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

public interface IAnalysisResult
{
}

/// <summary>
/// Result of the analysis pass.
/// </summary>
public interface IAnalysisResult<T>
{
    public T this[Expr expr]
    {
        get
        {
            if (TryGet(expr, out var value))
            {
                return value;
            }

            throw new InvalidOperationException($"Analysis not found for {expr}.");
        }
    }

    bool TryGet(Expr expr, [MaybeNullWhen(false)] out T value);
}
