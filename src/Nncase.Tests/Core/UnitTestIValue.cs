// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections;
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
        var b = (IEnumerable)a;
        Assert.Equal(NoneType.Default, a.Type);
        Assert.Throws<InvalidOperationException>(() => a.Count);
        Assert.Throws<InvalidOperationException>(() => a[0]);
        Assert.Throws<InvalidOperationException>(() => a.AsTensor());
        Assert.Throws<InvalidOperationException>(() => a.AsTensors());
        Assert.Throws<InvalidOperationException>(() => a.GetEnumerator());
        Assert.Throws<InvalidOperationException>(() => b.GetEnumerator());
    }

    [Fact]
    public void TestTensorValue()
    {
        var dims = new int[] { 1, 3, 16, 16 };
        var a = new TensorValue(Tensor.Ones<float>(dims));
        Assert.Equal(new TensorType(DataTypes.Float32, dims), a.Type);
        Assert.True(a.Count == 1);
    }

    [Fact]
    public void TestTensorValueIndex()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var a = new TensorValue(ones);
        Assert.Equal(a, a[0]);
        Assert.Throws<IndexOutOfRangeException>(() => a[1]);
    }

    [Fact]
    public void TestTensorValueAsTensor()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var a = new TensorValue(ones);
        Assert.Equal(ones, a.AsTensor());
    }

    [Fact]
    public void TestTensorValueAsTensors()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var a = new TensorValue(ones);
        Assert.Single(a.AsTensors());
        Assert.Equal(ones, a.AsTensors()[0]);
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

        Assert.Equal(a, b);
        Assert.Equal(a, c);
        Assert.NotEqual(a, d);
        Assert.True(a.Equals((object)b));
        Assert.True(a.Equals((object)c));
        Assert.False(a.Equals((object)d));
    }

    [Fact]
    public void TestTensorValueGetHashCode()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var a = new TensorValue(ones);
        Assert.Equal(HashCode.Combine(ones), a.GetHashCode());
    }

    [Fact]
    public void TestTupleValue()
    {
        var dims = new int[] { 1, 3, 16, 16 };
        var tensor1 = Tensor.Ones<float>(dims);
        var tensor2 = Tensor.Zeros<float>(dims);
        var tensors = new Tensor[] { tensor1, tensor2 };
        var a = Value.FromTensors(tensors);
        Assert.Equal(new TupleType(tensors.Select(x => new TensorType(x.ElementType, x.Shape))), a.Type);
        Assert.Equal(tensors.Length, a.Count);
    }

    [Fact]
    public void TestTupleValueIndex()
    {
        var dims = new int[] { 1, 3, 16, 16 };
        var tensor1 = Tensor.Ones<float>(dims);
        var tensor2 = Tensor.Zeros<float>(dims);
        var tensors = new Tensor[] { tensor1, tensor2 };
        var a = Value.FromTensors(tensors);
        Assert.Equal(new TensorValue(tensor1), a[0]);
        Assert.Equal(new TensorValue(tensor2), a[1]);
    }

    [Fact]
    public void TestTupleValueAsTensor()
    {
        var dims = new int[] { 1, 3, 16, 16 };
        var tensor1 = Tensor.Ones<float>(dims);
        var tensor2 = Tensor.Zeros<float>(dims);
        var tensors = new Tensor[] { tensor1, tensor2 };
        var a = Value.FromTensors(tensors);
        Assert.Throws<InvalidOperationException>(() => a.AsTensor());
    }

    [Fact]
    public void TestTupleValueAsTensors()
    {
        var dims = new int[] { 1, 3, 16, 16 };
        var tensor1 = Tensor.Ones<float>(dims);
        var tensor2 = Tensor.Zeros<float>(dims);
        var tensors = new Tensor[] { tensor1, tensor2 };
        var a = Value.FromTensors(tensors);
        Assert.Equal(tensor1, a.AsTensors()[0]);
        Assert.Equal(tensor2, a.AsTensors()[1]);
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
        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
        Assert.True(a.Equals((object)b));
        Assert.False(a.Equals((object)c));
    }

    [Fact]
    public void TestTupleValueGetHashCode()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var tensors = new Tensor[] { ones, ones };
        var values = tensors.Select(x => new TensorValue(x)).ToArray();
        var a = new TupleValue(values);
        Assert.Equal(HashCode.Combine(values), a.GetHashCode());
    }
}
