using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class MatMulEvaluator : IEvaluator<MatMul>
    {
        public static Const Visit(EvaluatorContext context, MatMul matMul)
        {
            var input = context.GetTorchArgument(matMul, MatMul.Input);
            var other = context.GetTorchArgument(matMul, MatMul.Other);
            return input.matmul(other).ToConst();
        }
    }
}