// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform;

/// <summary>
/// the static mutator can create the mutator in the tir pass.
/// </summary>
public static class Mutator
{
    /// <summary>
    /// Unroll Loop
    /// </summary>
    /// <returns></returns>
    public static Func<ExprMutator> UnRollLoop() => () => new Mutators.UnRollLoop();

    /// <summary>
    /// flatten the sequential
    /// </summary>
    /// <returns></returns>
    public static Func<ExprMutator> FlattenSequential() => () => new Mutators.FlattenSequential();
}

