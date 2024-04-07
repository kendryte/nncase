// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class Binary : CPUKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(Binary), 0, "input");

    public static readonly ParameterInfo Rhs = new(typeof(Binary), 1, "input");

    public static readonly ParameterInfo Output = new(typeof(Binary), 2, "output");

    public BinaryOp BinaryOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"BinaryOp.{BinaryOp}";
    }
}
