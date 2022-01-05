using static Tensorflow.Binding;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensorflow.Tensor VisitCumSum(CumSum cumSum)
        {
            var input = _context.GetTFArgument(cumSum, CumSum.Input);
            // in onnx, CumSum.Axis is a input tensor with one value
            var axis = _context.GetArgumentConst(cumSum, CumSum.Axis).ToTensor<int>()[0];
            var exclusive = _context.GetArgumentConstScalar<bool>(cumSum, CumSum.Exclusive);
            var reverse = _context.GetArgumentConstScalar<bool>(cumSum, CumSum.Reverse);
            return tf.cumsum(input, axis, exclusive, reverse);
        }
    }
}