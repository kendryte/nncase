// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class Binary : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(Binary), 0, "lhs");

    public static readonly ParameterInfo Rhs = new(typeof(Binary), 1, "rhs");

    public static readonly ParameterInfo Output = new(typeof(Binary), 2, "output");

    public BinaryOp BinaryOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"BinaryOp.{BinaryOp}";
    }
}
