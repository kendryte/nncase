// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// ir printer context interface.
/// </summary>
public interface IIRPrinterContext
{
    /// <summary>
    /// Get argument expression.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument expression.</returns>
    IPrintSymbol GetArgument(Op op, ParameterInfo parameter);

    /// <summary>
    /// get op
    /// </summary>
    /// <param name="op"></param>
    /// <returns></returns>
    IPrintSymbol Get(Op op);

    /// <summary>
    /// Get arguments expression.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <returns>The arguments expression.</returns>
    IPrintSymbol[] GetArguments(Op op);
}
