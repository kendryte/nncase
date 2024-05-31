// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.IR.Buffers;

/// <summary>
/// Buffer subview.
/// </summary>
public sealed partial class BufferSubview : Op
{
    /// <summary>
    /// Get the input parameter.
    /// </summary>
    public static readonly ParameterInfo Buffer = new(typeof(BufferSubview), 0, "buffer");

    /// <summary>
    /// Get the offset parameter.
    /// </summary>
    public static readonly ParameterInfo Offset = new(typeof(BufferSubview), 1, "offset");

    /// <summary>
    /// Get the shape parameter.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(BufferSubview), 2, "shape");

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
