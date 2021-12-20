using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitCelu(Celu celu)
        {
            var input = _context.GetArgument(celu, Celu.Input);
            return input.celu();
        }
    }
}