using Nncase.IR.Tensors;
using static Tensorflow.Binding;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class SqueezeEvaluator : IEvaluator<Squeeze>
    {
        private Const Visit(EvaluatorContext context, Squeeze squeeze)
        {
            var input = context.GetTFArgument(squeeze, Squeeze.Input);
            var dims = context.GetArgumentConst(squeeze, Squeeze.Dim).ToArray<int>();
            return tf.squeeze(input, dims).ToConst();
        }
    }
}