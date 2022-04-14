// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.NN;
using Nncase.IR.Random;
using Nncase.IR.Tensors;

namespace Nncase.IR.F;

/// <summary>
/// Random functional helper.
/// </summary>
public static class Random
{
    public static Call Normal(DataType type, Expr mean, Expr scale, Expr seed, Expr shape) =>
        new Call(new Normal(type), mean, scale, seed, shape);

    public static Call Normal(DataType type, Expr shape) => Normal(type, 0, 1, 0, shape);
    
    public static Call NormalLike(DataType type, Expr input, Expr mean, Expr scale, Expr seed) =>
        new Call(new NormalLike(type), input, mean, scale, seed);

    public static Call Uniform(DataType type, Expr high, Expr low, Expr seed, Expr shape) =>
        new Call(new Uniform(type), high, low, seed, shape);

    public static Call UniformLike(DataType type, Expr input, Expr high, Expr low, Expr seed) =>
        new Call(new UniformLike(type), input, high, low, seed);
}
