using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class UnSqueezeEvaluator : IEvaluator<UnSqueeze>
    {
        public Const Visit(EvaluatorContext context, UnSqueeze unSqueeze)
        {
            var input = context.GetTFArgument(unSqueeze, UnSqueeze.Input);
            var dims = context.GetArgumentConst(unSqueeze, UnSqueeze.Dim)
                .ToArray<int>()
                .Select(
                    x => Util.PositiveIndex(x, input.shape.rank
                    ))
                .ToArray();
            foreach (var dim in dims)
            {
                input = tf.expand_dims(input, Util.PositiveIndex(dim, input.shape.rank));
            }

            return input.ToConst();
        }
    }
}