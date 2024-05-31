// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using CommunityToolkit.HighPerformance.Helpers;
using Nncase;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestIValue
{
    public static IEnumerable<object[]> TestTensorValueCountData =>
        new[]
        {
            new object[] { Tensor.Ones<float>(new int[] { 1, 3, 16, 16 }) },
            new object[] { Tensor.Zeros<float>(new int[] { 1, 3, 16, 16 }) },
            new object[] { Tensor.From(new int[] { 1, 2, 3, 4 }, new int[] { 2, 2 }) },
        };

    public static IEnumerable<object[]> TestTupleValueCountData =>
        new[]
        {
            new object[] { new Tensor[] { Tensor.Ones<float>(new int[] { 1, 3, 16, 16 }) } },
            new object[] { new Tensor[] { Tensor.Ones<float>(new int[] { 1, 3, 16, 16 }), Tensor.Zeros<float>(new int[] { 1, 3, 16, 16 }) } },
            new object[] { new Tensor[] { Tensor.Ones<float>(new int[] { 1, 3, 16, 16 }), Tensor.Zeros<float>(new int[] { 1, 3, 16, 16 }), Tensor.From(new int[] { 1, 2, 3, 4 }, new int[] { 2, 2 }) } },
        };

    [Fact]
    public void TestNoneValueType()
    {
        var a = Value.None;
        Assert.Equal(NoneType.Default, a.Type);
    }

    [Fact]
    public void TestNoneValueCount()
    {
        var a = Value.None;
        Assert.True(a.Count == 1);
    }

    [Fact]
    public void TestNoneValueIndex()
    {
        var a = Value.None;
        Assert.Equal(a, a[0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => a[1]);
    }

    [Fact]
    public void TestNoneValueException()
    {
        var a = Value.None;
        Assert.Throws<InvalidOperationException>(() => a.AsTensor());
        Assert.Throws<InvalidOperationException>(() => a.AsTensors());
    }

    [Fact]
    public void TestTensorValueType()
    {
        var dims = new int[] { 1, 3, 16, 16 };
        var a = new TensorValue(Tensor.Ones<float>(dims));
        Assert.Equal(new TensorType(DataTypes.Float32, dims), a.Type);
    }

    [Theory]
    [MemberData(nameof(TestTensorValueCountData))]
    public void TestTensorValueCount(Tensor t)
    {
        var a = new TensorValue(t);
        Assert.True(a.Count == 1);
    }

    [Fact]
    public void TestTensorValueIndex()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var a = new TensorValue(ones);
        Assert.Equal(a, a[0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => a[1]);
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

        Assert.StrictEqual(a, b);
        Assert.StrictEqual(a, c);
        Assert.NotStrictEqual(a, d);
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
    public void TestTupleValueType()
    {
        var dims = new int[] { 1, 3, 16, 16 };
        var tensor1 = Tensor.Ones<float>(dims);
        var tensor2 = Tensor.Zeros<float>(dims);
        var tensors = new Tensor[] { tensor1, tensor2 };
        var a = Value.FromTensors(tensors);
        Assert.Equal(new TupleType(tensors.Select(x => new TensorType(x.ElementType, x.Shape))), a.Type);
    }

    [Theory]
    [MemberData(nameof(TestTupleValueCountData))]
    public void TestTupleValueCount(Tensor[] tensors)
    {
        var a = Value.FromTensors(tensors);
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
        Assert.Equal(a, c);
        Assert.True(a.Equals((object)b));
        Assert.True(a.Equals((object)c));
    }

    [Fact]
    public void TestTupleValueGetHashCode()
    {
        var ones = Tensor.Ones<float>(new int[] { 1, 3, 16, 16 });
        var tensors = new Tensor[] { ones, ones };
        var values = tensors.Select(x => new TensorValue(x)).ToArray();
        var a = new TupleValue(values);
        Assert.Equal(HashCode<TensorValue>.Combine(values), a.GetHashCode());
        Assert.Equal(a.ToString(), "(" + string.Join(",", values.Select(v => v.ToString())) + ")");
    }

    [Fact]
    public void TestNoneValue()
    {
        var defaultInstance = NoneValue.Default;
        Assert.NotNull(defaultInstance);

        var noneValue = NoneValue.Default;
        var type = noneValue.Type;
        Assert.Equal(NoneType.Default, type);

        var noneValue1 = NoneValue.Default;
        var count = noneValue1.Count;
        Assert.Equal(1, count);

        var noneValue2 = NoneValue.Default;
        var indexerResult = noneValue2[0];
        Assert.Equal(noneValue2, indexerResult);

        var noneValue3 = NoneValue.Default;
        Assert.Throws<ArgumentOutOfRangeException>(() => noneValue3[1]);

        var noneValue4 = NoneValue.Default;
        Assert.Throws<InvalidOperationException>(() => noneValue4.AsTensor());

        var noneValue5 = NoneValue.Default;
        using var enumerator = noneValue5.GetEnumerator();
        Assert.False(enumerator.MoveNext());

        var noneValue6 = NoneValue.Default;
        var other = NoneValue.Default;
        var equals = noneValue6.Equals(other);
        Assert.True(equals);

        var noneValue7 = NoneValue.Default;
        var other1 = None.Default;
        var equals1 = noneValue7.Equals(other1);
        Assert.False(equals1);
        Assert.Equal(0, noneValue7.GetHashCode());
    }
}
