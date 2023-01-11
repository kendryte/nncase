// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestIValue
{
    [Fact]
    public void TestNoneValue()
    {
        var a = Value.None;
        Assert.True(a.Type == NoneType.Default);
        Assert.Throws<InvalidOperationException>(() => a.Count);
        Assert.Throws<InvalidOperationException>(() => a[0]);
        Assert.Throws<InvalidOperationException>(() => a.AsTensor());
        Assert.Throws<InvalidOperationException>(() => a.AsTensors());
    }

    [Fact]
    public void TestTensorValue()
    {
        var a = new TensorValue(Tensor.Ones<float>(new int[] { 1, 3, 16, 16 }));
        Assert.True(a.Count == 1);
    }

    [Fact]
    public void TestTensorValueIndex()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var a = new TensorValue(ones);
        Assert.True(a[0] == a);
        Assert.Throws<IndexOutOfRangeException>(() => a[1]);
    }

    [Fact]
    public void TestTensorValueAsTensor()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var a = new TensorValue(ones);
        Assert.True(a.AsTensor() == ones);
    }

    [Fact]
    public void TestTensorValueAsTensors()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var a = new TensorValue(ones);
        Assert.True(a.AsTensors().Length == 1 && a.AsTensors()[0] == ones);
    }

    [Fact]
    public void TestTensorValueCompare()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var zeros = Tensor.Zeros<float>(new int[] { 1, 3, 16, 16 });
        var a = new TensorValue(ones);
        var b = a;
        var c = new TensorValue(ones);
        var d = new TensorValue(zeros);

        Assert.True(a == b);
        Assert.True(a != c);
        Assert.True(a != d);

        Assert.True(a.Equals(b));
        Assert.True(a.Equals(c));
        Assert.False(a.Equals(d));
        Assert.True(a.Equals((object)b));
        Assert.True(a.Equals((object)c));
        Assert.False(a.Equals((object)d));
    }

    [Fact]
    public void TestTupleValue()
    {
        var tensor1 = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var tensor2 = Tensor.Zeros<float>(new int[] { 1, 3, 16, 16 });
        var tensors = new Tensor[] { tensor1, tensor2 };
        var a = Value.FromTensors(tensors);
        Assert.True(a.Count == tensors.Length);
    }

    [Fact]
    public void TestTupleValueIndex()
    {
        var tensor1 = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var tensor2 = Tensor.Zeros<float>(new int[] { 1, 3, 16, 16 });
        var tensors = new Tensor[] { tensor1, tensor2 };
        var a = Value.FromTensors(tensors);
        Assert.True(a[0].Equals(new TensorValue(tensor1)));
        Assert.True(a[1].Equals(new TensorValue(tensor2)));
    }

    [Fact]
    public void TestTupleValueAsTensor()
    {
        var tensor1 = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var tensor2 = Tensor.Zeros<float>(new int[] { 1, 3, 16, 16 });
        var tensors = new Tensor[] { tensor1, tensor2 };
        var a = Value.FromTensors(tensors);
        Assert.Throws<InvalidOperationException>(() => a.AsTensor());
    }

    [Fact]
    public void TestTupleValueAsTensors()
    {
        var tensor1 = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var tensor2 = Tensor.Zeros<float>(new int[] { 1, 3, 16, 16 });
        var tensors = new Tensor[] { tensor1, tensor2 };
        var a = Value.FromTensors(tensors);
        Assert.True(a.AsTensors()[0] == tensor1 && a.AsTensors()[1] == tensor2);
    }

    [Fact]
    public void TestTupleValueCompare()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var zeros = Tensor.Zeros<float>(new int[] { 1, 3, 16, 16 });
        var tensors = new Tensor[] { ones, zeros };
        var a = Value.FromTensors(tensors);
        var b = a;
        var c = Value.FromTensors(new Tensor[] { ones, zeros });

        Assert.True(a == b);
        Assert.True(a != c);

        Assert.True(a.Equals(b));
        Assert.False(a.Equals(c));
        Assert.True(a.Equals((object)b));
        Assert.False(a.Equals((object)c));
    }
}
