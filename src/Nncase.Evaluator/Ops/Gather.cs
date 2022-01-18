using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class GatherEvaluator : IEvaluator<Gather>
    {
        public Const Visit(EvaluatorContext context, Gather gather)
        {
            var input = context.GetTFArgument(gather, Gather.Input);
            var axis = context.GetArgumentConst(gather, Gather.Axis).ToScalar<int>();
            var index = context.GetTFArgument(gather, Gather.Index);
            return tf.gather(input, index, axis: axis).ToConst();
        }
    }
}