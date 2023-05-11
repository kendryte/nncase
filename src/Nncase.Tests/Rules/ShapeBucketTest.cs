// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Nncase.Tests.TransformTest;
using Nncase.Utilities;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules;

[AutoSetupTestMethod(InitSession = true)]
public class ShapeBucketTest : TransformTestBase
{
    private readonly ITestOutputHelper _testOutputHelper;

    public ShapeBucketTest(ITestOutputHelper testOutputHelper)
    {
        _testOutputHelper = testOutputHelper;
    }

    // [Fact]
    // public void TestToFusion()
    // {
    //     var lhs = Testing.Rand<float>(3, 1, 24, 24);
    //     var rhs = Testing.Rand<float>(3, 1, 24, 24);
    //     var tr = Transpose(lhs, new[] { 1, 0, 2, 3 });
    //     var abs = Abs(rhs);
    //     var matmul = IR.F.Math.MatMul(tr, abs);
    //     var expr = Abs(matmul);
    //     TestMatched<MatmulToFusion>(expr);
    // }
    [Fact]
    public void TestBucket()
    {
        CompileOptions.DumpFlags = DumpFlags.Rewrite;
        var dim = 5;
        var inputA = Testing.Rand<float>(1, 3, 24, dim);
        var inputB = Testing.Rand<float>(1, 3, dim, 24);
        var effectVar = new Var("v", new TensorType(DataTypes.Int32, Shape.Scalar));
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 1, 3, 24, Dimension.Unknown }));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 24 }));


        var f = new VarFusion("stackvm", new[] { effectVar }, IR.F.Math.MatMul(lhs, rhs), lhs, rhs);

        var mainLhs = new Var("mainLhs", new TensorType(DataTypes.Float32, new[] { 1, 3, 24, Dimension.Unknown }));
        var mainRhs = new Var("mainRhs", new TensorType(DataTypes.Float32, new[] { 1, 3, Dimension.Unknown, 24 }));
        var mainInputInfo = new Dictionary<Var, Expr[]>
        {
            { mainLhs, new[] { 1, 3, 24, (Expr)effectVar } }, { mainRhs, new[] { 1, 3, (Expr)effectVar, 24 } },
        };

        var call = new Call(f, mainLhs, mainRhs);
        var main = new Function(call, mainLhs, mainRhs);
        var dict = new Dictionary<string, (int, int)> { { "v", (4, 8) } };
        Assert.True(main.InferenceType());
        Dumpper.DumpIR(main, "main");
        TestMatchedCore(main.Body, new Dictionary<Var, IValue>{{mainLhs, Value.FromTensor(inputA)}, {mainRhs, Value.FromTensor(inputB)}}, new[] { new FusionBucket(mainInputInfo, dict) });
    }

    [Fact]
    public void TestFindVar()
    {
        var v1 = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var v2 = new Var(new TensorType(DataTypes.Int32, Shape.Scalar));
        var expr = ((v1 * 4) + (v2 * 3)) * 2;
        var visitor = new FindVar();
        visitor.Visit(expr);
        Assert.Equal(visitor.Vars, new HashSet<Var>(new[] { v1, v2 }));
    }

    [Fact]
    public void TestGetBoundDict()
    {
        var batch = new Var("batch", new TensorType(DataTypes.Int32, Shape.Scalar));
        var tok_len = new Var("tok_len", new TensorType(DataTypes.Int32, Shape.Scalar));
        var enc_len = new Var("enc_len", new TensorType(DataTypes.Int32, Shape.Scalar));
        var dec_len = new Var("dec_len", new TensorType(DataTypes.Int32, Shape.Scalar));
        var tokens = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, Dimension.Unknown }));
        var enc_k = new Var(new TensorType(DataTypes.Float32, new[] { 3, Dimension.Unknown, 1, 256 }));
        var enc_v = new Var(new TensorType(DataTypes.Float32, new[] { 3, Dimension.Unknown, 1, 256 }));
        var enc_pad_mask = new Var(new TensorType(DataTypes.Float32, new[] { 1, Dimension.Unknown }));
        var dec_k1 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_k2 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_k3 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_v1 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_v2 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var dec_v3 = new Var(new TensorType(DataTypes.Float32, new[] { Dimension.Unknown, 4, Dimension.Unknown, 64 }));
        var inputInfo = new Dictionary<Var, Expr[]>
        {
            { tokens, new Expr[] { batch, tok_len } },
            { enc_k, new Expr[] { 3, enc_len, 1, 256 } },
            { enc_v, new Expr[] { 3, enc_len, 1, 256 } },
            { enc_pad_mask, new Expr[] { 1, enc_len } },
            { dec_k1, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_k2, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_k3, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_v1, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_v2, new Expr[] { batch, 4, dec_len, 64 } },
            { dec_v3, new Expr[] { batch, 4, dec_len, 64 } },
        };

        // todo: import相同名字的var归一化
        var dict = new Dictionary<string, (int, int)>
        {
            { "batch", (24, 48) }, { "tok_len", (24, 48) }, { "enc_len", (24, 48) }, { "dec_len", (24, 48) },
        };
        var (minDict, maxDict) = ReplaceRewrite.GetBoundDict(inputInfo, dict);
        Assert.Equal(Enumerable.Repeat(48, minDict.Count).ToArray(), maxDict.Values.Select(x => x.AsTensor().ToScalar<int>()));
        Assert.Equal(Enumerable.Repeat(24, maxDict.Count).ToArray(), minDict.Values.Select(x => x.AsTensor().ToScalar<int>()));
    }

    // public virtual Tensor[] MakeInputs(TensorType[] types)
    // {
    //     // return types.Select(type => Testing.Rand(type.DType, type.Shape.ToValueArray())).ToArray();
    //     var batch = 2;
    //     var tok_len = 3;
    //     var enc_len = 6;
    //     var dec_len = 2;
    //     var in0 = Testing.Rand<long>(batch, tok_len);
    //     var in1 = Testing.Rand<float>(3, enc_len, 1, 256);
    //     var in2 = Testing.Rand<float>(3, enc_len, 1, 256);
    //     var in3 = Testing.Rand<bool>(1, enc_len);
    //     var in4 = Testing.Rand<float>(batch, 4, dec_len, 64);
    //     var in5 = Testing.Rand<float>(batch, 4, dec_len, 64);
    //     var in6 = Testing.Rand<float>(batch, 4, dec_len, 64);
    //     var in7 = Testing.Rand<float>(batch, 4, dec_len, 64);
    //     var in8 = Testing.Rand<float>(batch, 4, dec_len, 64);
    //     var in9 = Testing.Rand<float>(batch, 4, dec_len, 64);
    //     return new[] { (Tensor)in0, in1, in2, in3, in4, in5, in6, in7, in8, in9 };
    // }

    [Fact]
    public void TestValue()
    {
        Expr a = new long[] { 9223372036854775807 };
        var value = Cast(a[0], DataTypes.Int32).Evaluate().AsTensor().ToScalar<long>();
        _testOutputHelper.WriteLine(value.ToString());
    }

    [Fact]
    public async Task TestSliceExpr()
    {
        var data = new Var(new TensorType(DataTypes.Int32, new[] { Dimension.Unknown, Dimension.Unknown }));
        var batch = Scalar("batch");
        var tok_len = Scalar("tok_len");
        var dict = new Dictionary<Var, Expr[]> { { data, new[] { batch, tok_len } } };
        var slice = Slice(data, new long[] { -1 }, new long[] { 9223372036854775807 }, new long[] { 1 }, new long[] { 1 });

        var shapeExpr = slice.EvaluateShapeExpr(dict);
        Dumpper.DumpIR(shapeExpr, "shapeExpr");
        var varValue = new Dictionary<Var, IValue>
        {
            { batch, Value.FromTensor(1) }, { tok_len, Value.FromTensor(1) },
        };
        var result = shapeExpr.Evaluate(varValue);
        _testOutputHelper.WriteLine(string.Join(",", result.AsTensor().ToArray<int>()));
    }

    private Var Scalar(string name) => new Var(new TensorType(DataTypes.Int32, Shape.Scalar));

    [Fact]
    public void TestComputeSegmentList()
    {
        var min = 3;
        var max = 9;
        var count = 2;
        var segments = ReplaceRewrite.ComputeSegmentList(count, min, max);
        Assert.Equal(segments, new[]{3, 9});
    }

    public Tensor[] MakeInputs(TensorType[] types)
    {
        var root = "/Users/homura/tmp/nmtv4_dec_fixed";
        var batch = 2;
        var tok_len = 3;
        var enc_len = 6;
        var dec_len = 2;
        var i = 2;
        var tokens = BinFileUtil.ReadBinFile(Path.Join(root, $"tokens_{i}.bin"), DataTypes.Int64, new[] { batch, tok_len });
        var enc_k = BinFileUtil.ReadBinFile(Path.Join(root, "enc_k.bin"), DataTypes.Float32, new[] { 3, enc_len, 1, 256 });
        var enc_v = BinFileUtil.ReadBinFile(Path.Join(root, "enc_v.bin"), DataTypes.Float32, new[] { 3, enc_len, 1, 256 });
        var enc_pad_mask = BinFileUtil.ReadBinFile(Path.Join(root, "enc_pad_mask.bin"), DataTypes.Boolean, new[] { 1, enc_len });
        var dec_k1 = BinFileUtil.ReadBinFile(Path.Join(root, $"dec_k_1_{i}.bin"), DataTypes.Float32,new[] { batch, 4, dec_len, 64 });
        var dec_k2 = BinFileUtil.ReadBinFile(Path.Join(root, $"dec_k_2_{i}.bin"), DataTypes.Float32,new[] { batch, 4, dec_len, 64 });
        var dec_k3 = BinFileUtil.ReadBinFile(Path.Join(root, $"dec_k_3_{i}.bin"), DataTypes.Float32,new[] { batch, 4, dec_len, 64 });
        var dec_v1 = BinFileUtil.ReadBinFile(Path.Join(root, $"dec_v_1_{i}.bin"), DataTypes.Float32,new[] { batch, 4, dec_len, 64 });
        var dec_v2 = BinFileUtil.ReadBinFile(Path.Join(root, $"dec_v_2_{i}.bin"), DataTypes.Float32,new[] { batch, 4, dec_len, 64 });
        var dec_v3 = BinFileUtil.ReadBinFile(Path.Join(root, $"dec_v_3_{i}.bin"), DataTypes.Float32,new[] { batch, 4, dec_len, 64 });
        return new[] { tokens, enc_k, enc_v, enc_pad_mask, dec_k1, dec_k2, dec_k3, dec_v1, dec_v2, dec_v3 };
    }
}
