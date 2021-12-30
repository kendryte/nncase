using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitSize(Size size)
        {
            var input = _context.GetTorchArgument(size, Size.Input);
            var v = (Const)((int)input.numel());
            return v.ToTorchTensor();
        }
    }
}