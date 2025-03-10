// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Diagnostics;

/// <summary>
/// Type inferencer interface.
/// </summary>
public interface IOpPrinter
{
    /// <summary>
    /// printer op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    string Visit(IPrintOpContext context, Op target);
}

/// <summary>
/// Type inferencer interface.
/// </summary>
public interface IOpPrinter<T> : IOpPrinter
    where T : Op
{
    /// <summary>
    /// Inference type of op.
    /// </summary>
    /// <param name="context">Context.</param>
    /// <param name="target">Target operator.</param>
    /// <returns>Result.</returns>
    string Visit(IPrintOpContext context, T target);

    string IOpPrinter.Visit(IPrintOpContext ctx, Op target)
    {
        return Visit(ctx, (T)target);
    }
}
