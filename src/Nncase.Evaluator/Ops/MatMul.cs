using Nncase.IR.Tensors;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitMatMul(MatMul matMul)
        {
            var input = _context.GetArgument(matMul, MatMul.Input);
            var other = _context.GetArgument(matMul, MatMul.Other);
            return input.matmul(other);
        }
    }
}