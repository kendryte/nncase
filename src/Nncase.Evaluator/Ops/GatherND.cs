using Nncase.IR.Tensors;
using Tensorflow;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensorflow.Tensor gather_nd(Tensor input, Tensor indices, Tensor batchDims)
        {
            return Binding.tf.Context.ExecuteOp("GatherNd", null,
                new ExecuteOpArgs(input, indices));
        }

        private Tensorflow.Tensor VisitGatherND(GatherND gatherND)
        {
            var input = _context.GetTFArgument(gatherND, GatherND.Input);
            var indices = _context.GetTFArgument(gatherND, GatherND.Index);
            var batchDims = _context.GetTFArgument(gatherND, GatherND.BatchDims);
            return gather_nd(input, indices, batchDims);
        }
    }
}