// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform.Mutators;

/// <summary>
/// This mutator only mutate the primfunc 
/// </summary>
public abstract class PrimFuncMutator : ExprMutator
{
    /// <inheritdoc/>
    /// shouldn't change the funciton
    public override Expr Visit(Function expr) => expr;

    /// <inheritdoc/>
    public override Expr Visit(Fusion expr) => expr;

    /// <inheritdoc/>
    public override Expr Visit(PrimFunctionWrapper expr) => expr;

}