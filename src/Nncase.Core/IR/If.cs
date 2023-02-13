// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// if(Condition) then { Then } else { Else }.
/// </summary>
public sealed record If(Expr Condition, Expr Then, Expr Else) : Expr
{
}
