// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;

namespace Nncase.Pattern.F;

/// <summary>
/// Math patterns.
/// </summary
public static partial class Math
{
    /// <summary>
    /// CallPattern unary.
    /// </summary>
    /// <param name = "unaryOp">Unary operator.</param>
    /// <param name = "expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static CallPattern Unary(UnaryOp unaryOp, ExprPattern input)
    {
        return new(new OpPattern<Unary>(x => (x.UnaryOp == unaryOp)), input);
    }
}
