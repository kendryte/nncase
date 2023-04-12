// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TIRTest;

public class UnitTestTBuffer
{
    [Fact]
    public void TestBuffer()
    {
        // var m = T.SizeVar("m");
        // var n = T.SizeVar("n");
        // var l = T.SizeVar("l");

        // var Ab = T.DeclBuffer((m, n), DataTypes.Float32);
        // var Bb = T.DeclBuffer((n, l), DataTypes.Float32);

        // Assert.IsType<Buffer>(Ab);
        // Assert.Equal(Ab.Dtype, DataTypes.Float32);
        // Assert.Equal(Ab.Shape[0], m);
        // Assert.Equal(Ab.Shape[1], n);
    }

    [Fact]
    public void TestBufferAccessPtr()
    {
        // var m = T.SizeVar("m");
        // var n = T.SizeVar("n");
        // var dict = new Dictionary<Var, IValue>() {
        //       { n, Value.FromTensor(1) },
        //       { m, Value.FromTensor(3) },
        //     };
        // var Ab = T.DeclBuffer((m, n),
        //               DataTypes.Float32,
        //               strides: (n + 1, 1));
        // var aptr = Ab.AccessPtr(AccessMode.ReadWrite);
        // Assert.Equal(aptr.Arguments[2].Evaluate(dict), (Ab.Strides[0] * m).Evaluate(dict));
        // Assert.IsType<AccessPtr>(aptr.Target);
    }

    [Fact]
    public void TestBufferAccessPtrOffset()
    {
        // var m = T.SizeVar("m");
        // var n = T.SizeVar("n");
        // var dict = new Dictionary<Var, torch.Tensor>() {
        //       { n,  torch.tensor(1) },
        //       { m,  torch.tensor(3) },
        //     };
        // var Ab = T.DeclBuffer((m, n),
        //               DataTypes.Float32,
        //               strides: (n + 1, 1));
        // var aptr = Ab.AccessPtr(AccessMode.ReadWrite, offset: 100);
        // Assert.Equal(AccessMode.ReadWrite, ((AccessPtr)aptr.Target).AccessMode);

        // var v = T.SizeVar("v");

        // var aptr2 = Ab.AccessPtr(AccessMode.ReadWrite, offset: 100 + 100 + v);

        // Testing.AssertExprEqual(aptr2.Arguments[1], v + 200);
    }

    // [Fact]
    // public void TestBufferAccessPtrExtent()
    // {
    //     var m = T.SizeVar("m");
    //     var n = T.SizeVar("n");
    //     var Ab = T.DeclBuffer((m, n), DataTypes.Float32);
    //     var aptr = Ab.AccessPtr(AccessMode.ReadWrite, offset: 100);
    //     Testing.AssertExprEqual(aptr.Arguments[2], m * n - 100);
    //     var Bb = T.DeclBuffer((m, n), DataTypes.Float32, strides: (n + 1, 1));
    //     var bptr = Bb.AccessPtr(AccessMode.ReadWrite, offset: 100);
    //     Testing.AssertExprEqual(bptr.Arguments[2], Bb.Strides[0] * m - 100);
    // }

    // [Fact]
    // public void TestBufferVLoad()
    // {
    //     var m = T.SizeVar("m");
    //     var n = T.SizeVar("n");
    //     var Ab = T.DeclBuffer((m, n), DataTypes.Float32, elem_offset: 100);
    //     var load = Ab.VLoad((2, 3));
    //     Testing.AssertExprEqual(load[Load.Index], 100 + ((2 * n) + 3));
    // }
}
