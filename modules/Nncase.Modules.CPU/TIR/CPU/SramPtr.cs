// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.CPU;

public sealed partial class SramPtr : Op
{
    public static readonly ParameterInfo OffSet = new(typeof(SramPtr), 0, "offset", IsIntegralScalar());

    public DataType DataType { get; }

    public override bool CanFoldConstCall => false;
}
