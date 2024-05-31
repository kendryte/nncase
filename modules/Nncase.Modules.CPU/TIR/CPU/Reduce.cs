// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class Reduce : CPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Reduce), 0, "input");

    public static readonly ParameterInfo InitValue = new(typeof(Reduce), 1, "initValue");

    public static readonly ParameterInfo Output = new(typeof(Reduce), 2, "output");

    public IRArray<int> PackedAxes { get; }

    public IRArray<int> PadedNums { get; }

    public Nncase.IR.IRArray<int> Axis { get; }

    public bool KeepDims { get; }

    public ReduceOp ReduceOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"ReduceOp.{ReduceOp}";
}
