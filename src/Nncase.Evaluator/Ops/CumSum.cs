using static Tensorflow.Binding;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Ops
{
    public class CumSumEvaluator : IEvaluator<CumSum>
    {
        public Const Visit(EvaluatorContext context, CumSum cumSum)
        {
            var input = context.GetTFArgument(cumSum, CumSum.Input);
            // in onnx, CumSum.Axis is a input tensor with one value
            var axis = context.GetArgumentConst(cumSum, CumSum.Axis).ToTensor<int>()[0];
            var exclusive = context.GetArgumentConstScalar<bool>(cumSum, CumSum.Exclusive);
            var reverse = context.GetArgumentConstScalar<bool>(cumSum, CumSum.Reverse);
            return tf.cumsum(input, axis, exclusive, reverse).ToConst();
        }
    }
}