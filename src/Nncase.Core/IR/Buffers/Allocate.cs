// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.IR.Buffers;

/// <summary>
/// get the buffer basement.
/// </summary>
public sealed partial class Allocate : Op
{
    /// <summary>
    /// Get the input parameter.
    /// </summary>
    public static readonly ParameterInfo Size = new(typeof(Allocate), 0, "size", TypePatternUtility.IsIntegralScalar());

    /// <summary>
    /// Get the alloacted buffer type.
    /// </summary>
    public DataType ElemType { get; }

    public TIR.MemoryLocation Location { get; }

    public override string DisplayProperty() => $"{ElemType}, {Location}";

    public override bool CanFoldConstCall => false;
}
