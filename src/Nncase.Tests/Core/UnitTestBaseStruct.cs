using Xunit;

namespace Nncase.Tests.CoreTest
{
    public class UnitTestBaseStruct
    {
        [Fact]
        public void TestValueRangeToTensor()
        {
            var range = ValueRange<float>.Full;
            Assert.Equal(
                range.ToTensor,
                Tensor.FromArray(new[] { range.Min, range.Max }));
        }
    }
}
