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
    /// get op.
    /// </summary>
    IPrintSymbol Get(Op op);

    /// <summary>
    /// Get arguments expression.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <returns>The arguments expression.</returns>
    IPrintSymbol[] GetArguments(Op op);

    /// <summary>
    /// we can explict visit the expr which not in the params.
    /// </summary>
    /// <param name="expr">give the expr.</param>
    /// <returns> symobl. </returns>
    IPrintSymbol Visit(Expr expr);

    /// <summary>
    /// get the default Serialize.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <returns>string.</returns>/
    string GetDefault(Op op) => $"{Get(op)}({string.Join(", ", op.Parameters.Select(p => p.Name + ": " + GetArgument(op, p).Serialize()))})";

    /// <summary>
    /// get indent string.
    /// </summary>
    string Indent();
}
