// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Targets;
using QuikGraph.Graphviz;
using Xunit;

namespace Nncase.Tests.AffineTest;

public static class FunctionSamples
{
    /// <summary>
    /// Tileflow default case.
    /// </summary>
    /// <returns>function.</returns>
    public static Function Get1WithTarget(string target)
    {
        Function func;
        {
            var a = new Var(new TensorType(DataTypes.Float32, new[] { 128, 256 }));
            var b = new Var(new TensorType(DataTypes.Float32, new[] { 256, 384 }));
            var c = IR.F.Tensors.MatMul(a, b);
            var d = IR.F.Math.Exp(c);
            var e = new Var(new TensorType(DataTypes.Float32, new[] { 384, 512 }));
            var f = IR.F.Tensors.MatMul(d, e);
            func = new("main", target, f, [a, b, e]);
        }

        return func;
    }

    public static Function Get1() => Get1WithTarget(CPUTarget.Kind);

    public static Function Get1Matmul()
    {
        Function func;
        {
            var a = new Var(new TensorType(DataTypes.Float32, new[] { 128, 256 }));
            var b = new Var(new TensorType(DataTypes.Float32, new[] { 256, 384 }));
            var c = IR.F.Tensors.MatMul(a, b);
            func = new("main", CPUTarget.Kind, c, [a, b]);
        }

        return func;
    }

    public static Function Get1Exp()
    {
        Function func;
        {
            var a = new Var(new TensorType(DataTypes.Float32, new[] { 128, 384 }));
            var d = IR.F.Math.Exp(a);
            func = new("main", CPUTarget.Kind, d, [a]);
        }

        return func;
    }

    /// <summary>
    /// Tileflow default case with vectorize M.
    /// </summary>
    /// <returns>function.</returns>
    public static Function Get1VectorizeMN()
    {
        Function func;
        {
            var a = new Var(new TensorType(DataTypes.Float32, new[] { 128, 256 }));
            var b = new Var(new TensorType(DataTypes.Float32, new[] { 256, 384 }));
            var c = IR.F.NTT.VectorizedMatMul(IR.F.Tensors.Pack(a, new[] { 4, 4 }, new[] { 0, 1 }), IR.F.Tensors.Pack(b, new[] { 4, 4 }, new[] { 0, 1 }), new[] { 0, 1 }, new[] { 0, 1 }, false, false, false);
            var d = IR.F.Math.Exp(c);
            var e = new Var(new TensorType(DataTypes.Float32, new[] { 384, 512 }));
            var f = IR.F.NTT.VectorizedMatMul(d, IR.F.Tensors.Pack(e, new[] { 4 }, new[] { 0 }), new[] { 0, 1 }, new[] { 0 }, false, false, false);
            func = new("main", CPUTarget.Kind, f, [a, b, e]);
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
            func = new IR.Function("main", CPUTarget.Kind, g, [a, b, d, f]);
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
            var d = IR.F.Math.Mul(c, new[] { 1.0f });
            var e = IR.F.Math.Sub(new[] { 1.5f }, d);
            func = new IR.Function("main", CPUTarget.Kind, e, [a]);
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
            func = new IR.Function("main", CPUTarget.Kind, d, [a, b]);
        }

        return func;
    }

    /// <summary>
    /// the tuple output.
    /// </summary>
    public static Function Get5()
    {
        Function func;
        {
            var shape = new[] { 1, 12, 14, 14 };
            var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, shape));
            var b = new IR.Var("b", new IR.TensorType(DataTypes.Float32, shape));
            var c = IR.F.Math.Binary(BinaryOp.Mul, a, b);
            var d = IR.F.Math.Neg(c);
            func = new IR.Function("main", CPUTarget.Kind, new IR.Tuple(c, d), [a, b]);
        }

        return func;
    }

    /// <summary>
    /// get single op for mcts.
    /// </summary>
    public static Function Get6()
    {
        Function func;
        {
            var shape = new[] { 1, 12, 14, 14 };
            var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, shape));
            var b = new IR.Var("b", new IR.TensorType(DataTypes.Float32, shape));
            var c = IR.F.Math.Binary(BinaryOp.Mul, a, b);
            func = new IR.Function("main", CPUTarget.Kind, c, [a, b]);
        }

        return func;
    }
}
