using Nncase.IR.Tensors;
using static Tensorflow.Binding;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensorflow.Tensor VisitSqueeze(Squeeze squeeze)
        {
            var input = _context.GetTFArgument(squeeze, Squeeze.Input);
            var dims = _context.GetArgumentConst(squeeze, Squeeze.Dim).ToArray<int>();
            return tf.squeeze(input, dims);
        }
    }
}