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
        private torch.Tensor VisitBatchNormalization(BatchNormalization batchNorm)
        {
            var input = _context.GetArgument(batchNorm, BatchNormalization.Input);
            var eps = _context.GetArgumentConst(batchNorm, BatchNormalization.Epsilon);
            var mom = _context.GetArgumentConst(batchNorm, BatchNormalization.Momentum);
            var m = torch.nn.BatchNorm2d(input.shape[^3], eps.ToScalar<float>(), mom.ToScalar<float>());
            return m.forward(input);
        }
    }
}