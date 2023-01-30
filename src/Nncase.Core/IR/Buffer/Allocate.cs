// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.IR.Buffer;

/// <summary>
/// get the buffer basement.
/// </summary>
public record Allocate(TensorType ElemType) : Op
{
}
