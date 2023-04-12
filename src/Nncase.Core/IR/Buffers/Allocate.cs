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
    public TensorType ElemType { get; }
}
