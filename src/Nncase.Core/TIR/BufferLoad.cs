// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// Buffer load node.
/// </summary>
/// <param name="Buffer">The buffer to be loaded.</param>
/// <param name="Indices">The buffer indices.</param>
public sealed record BufferLoad(PhysicalBuffer Buffer, IRArray<Expr> Indices) : Expr
{
}
