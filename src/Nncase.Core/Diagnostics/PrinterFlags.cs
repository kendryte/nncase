// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;

namespace Nncase.Diagnostics;

[Flags]
public enum PrinterFlags
{
    /// <summary>
    /// printer stack depth less than 1.
    /// </summary>
    Minimal = 1 << 0,

    /// <summary>
    /// printer stack depth less than 4.
    /// </summary>
    Normal = 1 << 1,

    /// <summary>
    /// printer stack depth less than inf.
    /// </summary>
    Detailed = 1 << 2,

    /// <summary>
    /// print in inline mode.
    /// </summary>
    Inline = 1 << 3,

    /// <summary>
    /// print in script mode.
    /// </summary>
    Script = 1 << 4,

    /// <summary>
    /// skip dimension expr.
    /// </summary>
    SkipDimensionExpr = 1 << 5,
}
