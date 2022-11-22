using Nncase;
using Xunit;
namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTensor
{
    [Fact]
    public void TestIsContiguousSlice()
    {
        var dim1 = new[] { 1, 512, 14, 14 };

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..512, 0..14, 0..14 }));

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..1, 0..1, 0..14 }));

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..1, 0..1, 7..14 }));

        Assert.True(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..1, 7..14, 0..14 }));

        Assert.False(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..512, 0..7, 0..14 }));

        Assert.False(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 10..512, 0..1, 0..1 }));

        Assert.False(TensorUtilities.IsContiguousSlice(
          dim1,
          new[] { 0..1, 0..512, 0..7, 0..1 }));

        var dim2 = new[] { 1, 512, 1, 196 };

        Assert.True(TensorUtilities.IsContiguousSlice(
              dim2,
              new[] { 0..1, 0..128, 0..1, 0..196 }));

        Assert.True(TensorUtilities.IsContiguousSlice(
              dim2,
              new[] { 0..1, 0..1, 0..1, 10..15 }));
    }
}