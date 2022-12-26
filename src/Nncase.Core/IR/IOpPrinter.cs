// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

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
    /// <param name="iLmode">if is print is il or script.</param>
    /// <returns>Result.</returns>
    string Visit(IIRPrinterContext context, Op target, bool iLmode);
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
    /// <param name="iLmode">if is print like il.</param>
    /// <returns>Result.</returns>
    string Visit(IIRPrinterContext context, T target, bool iLmode);

    string IOpPrinter.Visit(IIRPrinterContext ctx, Op target, bool iLmode)
    {
        return Visit(ctx, (T)target, iLmode);
    }
}
