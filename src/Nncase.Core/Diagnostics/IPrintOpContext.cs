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
/// ir printer context interface.
/// </summary>
public interface IPrintOpContext
{
    /// <summary>
    /// Gets the print flags.
    /// </summary>
    PrinterFlags Flags { get; }

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
    /// get the default Serialize.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <returns>string.</returns>/
    string GetDefault(Op op)
    {
        var prop = op.DisplayProperty();
        var sep = prop.Length > 0 ? "," : string.Empty;
        return $"{Get(op)}({prop}{sep}{string.Join(", ", op.Parameters.Select(p => GetArgument(op, p).ToString()))})";
    }

    /// <summary>
    /// get indent string.
    /// </summary>
    string Indent();
}
