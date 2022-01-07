using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensorflow.Tensor VisitUnSqueeze(UnSqueeze unSqueeze)
        {
            var input = _context.GetTFArgument(unSqueeze, UnSqueeze.Input);
            var dims = _context.GetArgumentConst(unSqueeze, UnSqueeze.Dim)
                .ToArray<int>()
                .Select(
                    x => Util.PositiveIndex(x, input.shape.rank
                    ))
                .ToArray();
            foreach (var dim in dims)
            {
                input = tf.expand_dims(input, Util.PositiveIndex(dim, input.shape.rank));
            }

            return input;
        }
    }
}