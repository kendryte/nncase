// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class Reduce : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Reduce), 0, "input");

    // TODO: support init value
    // public static readonly ParameterInfo InitValue = new(typeof(Reduce), 1, "initValue");
    public static readonly ParameterInfo Output = new(typeof(Reduce), 1, "output");

    public static readonly ParameterInfo LoadPrevious = new(typeof(Reduce), 2, "loadPrevious");

    public IRArray<int> VectorizedAxes { get; }

    public IRArray<Dimension> PadedNums { get; }

    public IRArray<int> Axes { get; }

    public bool KeepDims { get; }

    public ReduceOp ReduceOp { get; }

    /// <inheritdoc/>
    public override string DisplayProperty() => $"ReduceOp.{ReduceOp}";
}
