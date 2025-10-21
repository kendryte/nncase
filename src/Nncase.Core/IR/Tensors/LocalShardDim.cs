// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.ComponentModel;
using Nncase.IR;

namespace Nncase.IR.Tensors;

public sealed partial class LocalShardDim : Op
{
    public static readonly ParameterInfo Dim = new(typeof(LocalShardDim), 0, "dim");

    [Browsable(false)]
    public SBP AxisPolicy { get; }

    [Browsable(false)]
    public Placement Placement { get; }

    public override string DisplayProperty() => $"{nameof(AxisPolicy)}: {AxisPolicy}, {nameof(Placement)}: {Placement}";
}
