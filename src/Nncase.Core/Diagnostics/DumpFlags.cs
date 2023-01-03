// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Diagnostics;

/// <summary>
/// Dump flags.
/// </summary>
[Flags]
public enum DumpFlags
{
    /// <summary>
    /// Nothing need to be dump.
    /// </summary>
    None = 0,

    /// <summary>
    /// Dump import ops.
    /// </summary>
    ImportOps = 1,

    /// <summary>
    /// Dump pass pre and post ir.
    /// </summary>
    PassIR = 2,

    /// <summary>
    /// Dump egraph costs.
    /// </summary>
    EGraphCost = 4,

    /// <summary>
    /// Dump rewrite.
    /// </summary>
    Rewrite = 8,

    /// <summary>
    /// Dump calibration.
    /// </summary>
    Calibration = 16,
}
