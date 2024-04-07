// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class PtrOf : Op
{
    public string PtrName { get; }

    public DataType DataType { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"{PtrName}";
}
