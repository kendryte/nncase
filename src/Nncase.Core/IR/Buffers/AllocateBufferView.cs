// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.IR.Buffers;

/// <summary>
/// Allocate buffer view.
/// </summary>
public sealed partial class AllocateBufferView : Op
{
    /// <summary>
    /// Get the input parameter.
    /// </summary>
    public static readonly ParameterInfo Buffer = new(typeof(AllocateBufferView), 0, "buffer");

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
