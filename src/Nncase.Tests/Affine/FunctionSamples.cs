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
    public static Function GetMatmulExpMatmulWithTarget(string target)
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

    public static Function GetMatmulExpMatmul() => GetMatmulExpMatmulWithTarget(CPUTarget.Kind);

    public static Function GetMatmul()
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

    public static Function GetExp()
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
    public static Function GetVectorizeMatmulExpMatmul()
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

    public static Function GetMatmulBinaryBinary()
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

    public static Function GetMulDivMulSub()
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

    public static Function GetAddBranchMerge()
    {
        Function func;
        {
            var shape = new[] { 1, 128, 64, 32 };
            var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, shape));
            var a1 = IR.F.Math.Neg(a);
            var b1 = IR.F.Math.Sin(a1);
            var b2 = IR.F.Math.Cos(a1);
            var c = IR.F.Math.Add(b1, b2);
            var d = IR.F.Math.Square(c);
            func = new IR.Function("main", CPUTarget.Kind, d, [a]);
        }

        return func;
    }

    /// <summary>
    /// the tuple output.
    /// </summary>
    public static Function GetBinaryNeg()
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
    public static Function GetSingleBinary()
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

    /// <summary>
    /// qwen case.
    /// </summary>
    public static Function GetUnaryCastTrans()
    {
        Function func;
        {
            var seq_len = new DimVar("sequence_length")
            {
                Metadata = new()
                {
                    Range = new(1, 256),
                },
            };
            var shape = new RankedShape(new Dimension[] { seq_len, 128 });
            var v0 = new Var(new TensorType(DataTypes.Float32, shape));
            var v1 = IR.F.Math.Unary(UnaryOp.Sin, v0);
            var v2 = IR.F.Tensors.Cast(v1, DataTypes.Float16);
            var v3 = IR.F.Tensors.Transpose(v2, [1, 0]);
            new Passes.Transforms.InferRangeVisitor().Visit(v3);
            func = new("main", CPUTarget.Kind, v3, [v0]);
        }

        return func;
    }

    /// <summary>
    /// for check reconstruct result.
    /// </summary>
    public static Function GetBinaryUnary()
    {
        Function func;
        {
            var shape = new[] { 1, 12, 14, 14 };
            var a = new IR.Var("a", new IR.TensorType(DataTypes.Float32, shape));
            var b = new IR.Var("b", new IR.TensorType(DataTypes.Float32, shape));
            var c = IR.F.Math.Binary(BinaryOp.Mul, a, b);
            var d = IR.F.Math.Unary(UnaryOp.Neg, c);
            func = new IR.Function("main", CPUTarget.Kind, new IR.Tuple(c, d), [a, b]);
        }

        return func;
    }

    /// <summary>
    /// qwen case.
    /// </summary>
    public static Function GetQwen3Rope()
    {
        Function func;
        {
            var seq_len = new DimVar("sequence_length")
            {
                Metadata = new()
                {
                    Range = new(1, 256),
                },
            };

            // %11 is the input: use a Float16 var with shape [128, sequence_length]
            var v0 = new Var(new TensorType(DataTypes.Float16, new RankedShape(new Dimension[] { 128, seq_len })));

            // %12 = Transpose(%11, [1,0])
            var v12 = IR.F.Tensors.Transpose(v0, [1, 0]);

            // %13 = Reshape(%12, [sequence_length,64,2])
            var v13 = IR.F.Tensors.Reshape(v12, new Dimension[] { seq_len, 64, 2 });

            // %14 = VectorizedLayerNorm(..., %13, const(...[2]), const(...[2]), [0])
            // gamma = [1,1], beta = [0,0] in f16
            var gamma = IR.F.Tensors.Cast(new[] { 1.0f, 1.0f }, DataTypes.Float16);
            var beta = IR.F.Tensors.Cast(new[] { 0.0f, 0.0f }, DataTypes.Float16);

            // var v14 = IR.F.NTT.VectorizedLayerNorm(v13, axis: 2, epsilon: 1e-6f, useMean: false, vectorizedAxes: new[] { 2 }, gamma, beta, new[] { 0 });
            // Fallback:
            var v14 = IR.F.NN.LayerNorm(2, 1e-6f, v13, gamma, beta, false);

            // %15 = Transpose(%14, [1,2,0]) -> [64,2,sequence_length]
            // var v15 = IR.F.Tensors.Transpose(v14, [1, 2, 0]);
            var v15 = v14; // sequence_length, head, dim

            // %16 = GetPositionIds(sequence_length, ...)
            // Use a range [0..seq_len) as position ids
            var v16 = IR.F.Tensors.ConstantOfShape(new RankedShape(seq_len), 1.0f);

            // %17 = Reshape(%16, [sequence_length,1])
            var v17 = IR.F.Tensors.Reshape(v16, new[] { -1, 1 });

            // %18 = Mul(%17, const([4]))
            // Provide a 4-element frequency vector (placeholder values)
            var invFreq = new[] { 1.0f, 1.0f, 1.0f, 1.0f };
            var v18 = IR.F.Math.Binary(BinaryOp.Mul, v17, invFreq);

            // %19 = Cos(%18)
            var v19 = IR.F.Math.Unary(UnaryOp.Cos, v18);

            // %20 = VectorizedCast(f16<64>, ... , %19, None)
            var v20 = IR.F.Tensors.Cast(v19, DataTypes.Float16);

            // %21 = Transpose(%20, [1,0]) -> [2,sequence_length]
            // var v21 = IR.F.Tensors.Transpose(v20, [1, 0]);
            var v21 = v20; // [sequence_length, dim]

            // %22 = Sin(%18)
            var v22 = IR.F.Math.Unary(UnaryOp.Sin, v18);

            // %23 = VectorizedCast(f16<64>, ... , %22, None)
            var v23 = IR.F.Tensors.Cast(v22, DataTypes.Float16);

            // %24 = Transpose(%23, [1,0]) -> [2,sequence_length]
            // var v24 = IR.F.Tensors.Transpose(v23, [1, 0]);
            var v24 = v23; // [sequence_length, dim]

            // %25 = VectorizedRoPE(%15, %21, %24)
            var v25 = IR.F.NN.RoPE(v15, v21, v24);

            new Passes.Transforms.InferRangeVisitor().Visit(v25);
            func = new("main", CPUTarget.Kind, v25, [v0]);
        }

        return func;
    }

    public static Function GetDynamicVectorizedSwish()
    {
        var dim2 = new DimVar("dim2")
        {
            Metadata = new()
            {
                Range = new(1, 128),
            },
        };

        var dim3 = new DimVar("dim3")
        {
            Metadata = new()
            {
                Range = new(1, 128),
            },
        };

        var v1 = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(32, 12, 4 * Dimension.CeilDiv(dim2, 4), dim3)));
        var v2 = IR.F.Tensors.Pack(v1, [4], [2]); // {f32<4>[32,12,(ceil(dim2 / 4)),dim3], (S(0),B,B,B), [8@t,12,32,128]} ,
        var v3 = IR.F.NN.Swish(v2); // {f32<4>[32,12,(ceil(dim2 / 4)),dim3], (S(0),B,B,B), [8@t,12,32,128]} ,
        var v4 = IR.F.Tensors.Unpack(v3, [4], [2]); // {f32[32,12,(4 * ceil(dim2 / 4)),dim3], (S(0),B,B,B), [8@t,12,128,128]} ,
        return new Function("main", CPUTarget.Kind, v4, [v1, dim2, dim3]);
    }

    public static Function GetDynamicVectorizedCastTranspose()
    {
        var seq_len = new DimVar("seq_len")
        {
            Metadata = new()
            {
                Range = new(1, 256),
            },
        };

        var v1 = new Var("input", new TensorType(new VectorType(DataTypes.Float16, 64), new RankedShape(seq_len, 16)));
        var v2 = IR.F.NTT.VectorizedCast(v1, new VectorType(DataTypes.Float32, 32), CastMode.KDefault, [1], None.Default);
        var v3 = IR.F.Tensors.Transpose(v2, [1, 0]);
        return new Function("main", CPUTarget.Kind, v3, [v1]);
    }
}
