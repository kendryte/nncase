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
/// Buffer store node.
/// </summary>
/// <param name="Buffer">The buffer.</param>
/// <param name="Indices">The value we to be stored.</param>
/// <param name="Value">The indices location to be stored.</param>
public sealed record BufferStore(PhysicalBuffer Buffer, IRArray<Expr> Indices, Expr Value) : Expr
{
}
