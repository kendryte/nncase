// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.CPU;
using Nncase.IR.Math;

namespace Nncase.IR.F;

public partial class CPU
{
    /// <summary>
    /// Call unary.
    /// </summary>
    /// <param name="unaryOp">Unary operator.</param>
    /// <param name="expr">Source expression.</param>
    /// <returns>Result expression.</returns>
    public static Call CPUUnary(UnaryOp unaryOp, Expr expr)
    {
        return new Call(new CPUUnary(unaryOp), expr);
    }
}
