using Nncase.IR.Tensors;
using Tensorflow;
using static Tensorflow.Binding;
using Nncase.IR;

namespace Nncase.Evaluator.Ops
{
    public class GatherNDEvaluator : IEvaluator<GatherND>
    {
        private Tensorflow.Tensor gather_nd(Tensor input, Tensor indices, Tensor batchDims)
        {
            return Binding.tf.Context.ExecuteOp("GatherNd", null,
                new ExecuteOpArgs(input, indices));
        }

        private Const Visit(EvaluatorContext context, GatherND gatherND)
        {
            var input = context.GetTFArgument(gatherND, GatherND.Input);
            var indices = context.GetTFArgument(gatherND, GatherND.Index);
            var batchDims = context.GetTFArgument(gatherND, GatherND.BatchDims);
            return gather_nd(input, indices, batchDims).ToConst();
        }
    }
}