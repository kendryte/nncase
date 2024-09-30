// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class Compare : CPUKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(Compare), 0, "lhs");

    public static readonly ParameterInfo Rhs = new(typeof(Compare), 1, "rhs");

    public static readonly ParameterInfo Output = new(typeof(Compare), 2, "output");

    public CompareOp CompareOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"CompareOp.{CompareOp}";
    }
}
