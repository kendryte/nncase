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
/// Let binding. Bind var to value then evaluate body. return unit.
/// </summary>
/// <param name="Var"> The expr . </param>
/// <param name="Expression"> The value to be binded. </param>
/// <param name="Body"> The Let body. </param>
public sealed record Let(Var Var, Expr Expression, Sequential Body) : Expr
{
}
