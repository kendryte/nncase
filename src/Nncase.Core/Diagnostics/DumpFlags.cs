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
    ImportOps = 1 << 1,

    /// <summary>
    /// Dump pass pre and post ir.
    /// </summary>
    PassIR = 1 << 2,

    /// <summary>
    /// Dump egraph costs.
    /// </summary>
    EGraphCost = 1 << 3,

    /// <summary>
    /// Dump rewrite.
    /// </summary>
    Rewrite = 1 << 4,

    /// <summary>
    /// Dump calibration.
    /// </summary>
    Calibration = 1 << 5,

    /// <summary>
    /// Dump evaluator values.
    /// </summary>
    Evaluator = 1 << 6,

    /// <summary>
    /// Dump compile stages.
    /// </summary>
    Compile = 1 << 7,

    /// <summary>
    /// Dump tiling.
    /// </summary>
    Tiling = 1 << 8,

    /// <summary>
    /// Dump schedule.
    /// </summary>
    Schedule = 1 << 9,

    /// <summary>
    /// Dump codegen.
    /// </summary>
    CodeGen = 1 << 10,
}
