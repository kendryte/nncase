// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Schedule.TileTree;
using QuikGraph.Graphviz;
using Xunit;

namespace Nncase.Tests.AffineTest;

public static class FunctionSamples
{
    /// <summary>
    /// Tileflow default case.
    /// </summary>
    /// <returns>function.</returns>
    public static Function Get1()
    {
        Function func;
        {
            var a = new Var(new TensorType(DataTypes.Float32, new[] { 128, 256 }));
            var b = new Var(new TensorType(DataTypes.Float32, new[] { 256, 384 }));
            var c = IR.F.Tensors.MatMul(a, b);
            var d = IR.F.Math.Exp(c);
            var e = new Var(new TensorType(DataTypes.Float32, new[] { 384, 512 }));
            var f = IR.F.Tensors.MatMul(d, e);
            func = new(f, a, b, e);
        }

        return func;
    }

    /// <summary>
    /// Tileflow default case with pack M.
    /// </summary>
    /// <returns>function.</returns>
    public static Function Get1PackMN()
    {
        Function func;
        {
            var a = new Var(new TensorType(DataTypes.Float32, new[] { 128, 256 }));
            var b = new Var(new TensorType(DataTypes.Float32, new[] { 256, 384 }));
            var c = IR.F.CPU.PackedMatMul(IR.F.CPU.Pack(a, new[] { 4, 4 }, new[] { 0, 1 }), IR.F.CPU.Pack(b, new[] { 4, 4 }, new[] { 0, 1 }), new[] { 0, 1 }, Array.Empty<int>(), new[] { 0, 1 }, Array.Empty<int>());
            var d = IR.F.Math.Exp(c);
            var e = new Var(new TensorType(DataTypes.Float32, new[] { 384, 512 }));
            var f = IR.F.CPU.PackedMatMul(d, IR.F.CPU.Pack(e, new[] { 4 }, new[] { 0 }), new[] { 0, 1 }, Array.Empty<int>(), new[] { 0 }, Array.Empty<int>());
            func = new(f, a, b, e);
        }

        return func;
    }

    public static Function Get2()
    {
        Function func;
        {
            var ashape = new[] { 1, 64, 384, 128 };
            var bshape = new[] { 1, 64, 128, 384 };
            var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, ashape));
            var b = new IR.Var("b", new IR.TensorType(DataTypes.Float32, bshape));
            var c = IR.F.Tensors.MatMul(a, b);
            var dshape = new[] { 1 };
            var d = new IR.Var("d", new IR.TensorType(DataTypes.Float32, dshape));
            var e = IR.F.Math.Binary(BinaryOp.Div, c, d);
            var fshape = new[] { 1, 1, 384, 384 };
            var f = new IR.Var("f", new IR.TensorType(DataTypes.Float32, fshape));
            var g = IR.F.Math.Binary(BinaryOp.Add, e, f);
            func = new IR.Function("main", g, a, b, d, f);
        }

        return func;
    }

    public static Function Get3()
    {
        Function func;
        {
            var shape = new[] { 1, 12, 14, 14 };
            var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, shape));
            var b = IR.F.Math.Mul(a, new[] { 1.0f });
            var c = IR.F.Math.Div(b, new[] { 2.0f });
            var d = IR.F.Math.Mul(c, new[] { 3.0f });
            var e = IR.F.Math.Sub(new[] { 1.5f }, d);
            func = new IR.Function("main", e, a);
        }

        return func;
    }

    public static Function Get4()
    {
        Function func;
        {
            var shape = new[] { 1, 12, 14, 14 };
            var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, shape));
            var b = new IR.Var("b", new IR.TensorType(DataTypes.Float32, shape));
            var a1 = IR.F.Math.Neg(a);
            var b1 = IR.F.Math.Neg(b);
            var c = IR.F.Math.Add(a1, b1);
            var d = IR.F.Math.Neg(c);
            func = new IR.Function("main", d, a, b);
        }

        return func;
    }
}
