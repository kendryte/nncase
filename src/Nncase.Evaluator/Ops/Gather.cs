using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensorflow.Tensor VisitGather(Gather gather)
        {
            var input = _context.GetTFArgument(gather, Gather.Input);
            var axis = _context.GetArgumentConst(gather, Gather.Axis).ToScalar<int>();
            var index = _context.GetTFArgument(gather, Gather.Index);
            return tf.gather(input, index, axis: axis);
        }
    }
}