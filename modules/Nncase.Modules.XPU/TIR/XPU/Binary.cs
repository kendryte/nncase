// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.XPU;

public sealed partial class Binary : XPUKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(Binary), 0, "input");

    public static readonly ParameterInfo Rhs = new(typeof(Binary), 1, "input");

    public static readonly ParameterInfo Output = new(typeof(Binary), 2, "output");

    public BinaryOp BinaryOp { get; }

    public DistributedType LhsType { get; }

    public DistributedType RhsType { get; }

    public DistributedType OutType { get; }

    /// <inheritdoc/>
    public override string DisplayProperty()
    {
        return $"BinaryOp.{BinaryOp}";
    }
}
