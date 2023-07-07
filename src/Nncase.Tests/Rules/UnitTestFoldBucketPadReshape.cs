using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.Rules
{
    [AutoSetupTestMethod(InitSession = true)]
    public class UnitTestFoldBucketPadReshape : TransformTestBase
    {
        Call BucketPad(Expr input, Expr shape) => new Call(new BucketPad(), input, shape);

        [Fact]
        public void TestFoldBucketPadReshape()
        {
            var inputData = Testing.Rand<float>(1, 3, 12, 12);
            var input = new Var(new TensorType(inputData.ElementType, inputData.Shape));
            var reshape = Reshape(BucketPad(input, new[] { 1, 3, 24, 24 }), new[] { 1, 1, 3, 24, 24 });
            TestMatched<FoldBucketPadReshape>(reshape, new Dictionary<Var, IValue> { { input, Value.FromTensor(inputData) } });
        }

        [Fact]
        public void TestFoldBucketPadUnsqueeze()
        {
            var inputData = Testing.Rand<float>(1, 3, 12, 12);
            var input = new Var(new TensorType(inputData.ElementType, inputData.Shape));
            var reshape = Unsqueeze(BucketPad(input, new[] { 1, 3, 24, 24 }), new[] { 4 });
            TestMatched<FoldBucketPadUnsqueeze>(reshape, new Dictionary<Var, IValue> { { input, Value.FromTensor(inputData) } });
        }
    }
}
