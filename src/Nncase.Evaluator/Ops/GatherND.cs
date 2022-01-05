using Nncase.IR.Tensors;
using Tensorflow;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private Tensorflow.Tensor gather_nd(Tensor @params, Tensor indices, int batch_dims)
        {
            return Binding.tf.Context.ExecuteOp("GatherND", null, new ExecuteOpArgs(new object[]
            {
                (object) @params,
                (object) indices,
            }).SetAttributes((object) new
            {
                batch_dims = batch_dims
            }))[0];
        }
        
        private Tensorflow.Tensor VisitGatherND(GatherND gatherND)
        {
            var input = _context.GetTFArgument(gatherND, GatherND.Input);
            var indices = _context.GetTFArgument(gatherND, GatherND.Index);
            var batchDims = _context.GetArgumentConst(gatherND, GatherND.BatchDims).ToScalar<int>();
            return gather_nd(input, indices, batchDims);
        }
    }
}