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
            var input = _context.GetTorchArgument(batchNorm, BatchNormalization.Input);
            var eps = _context.GetArgumentConst(batchNorm, BatchNormalization.Epsilon);
            var mom = _context.GetArgumentConst(batchNorm, BatchNormalization.Momentum);
            var m = torch.nn.BatchNorm2d(input.shape[^3], eps.ToScalar<float>(), mom.ToScalar<float>());
            return m.forward(input);
        }

        private torch.Tensor VisitInstanceNormalization(InstanceNormalization i)
        {
            var input = _context.GetTorchArgument(i, InstanceNormalization.Input);
            var eps = _context.GetArgumentConst(i, InstanceNormalization.Epsilon).ToScalar<float>();
            var f = torch.nn.InstanceNorm2d(input.shape[1], eps);
            return f.forward(input);
        }

        private torch.Tensor VisitLRN(LRN l)
        {
            var input = _context.GetTorchArgument(l, LRN.Input);
            var size = _context.GetArgumentConstScalar<long>(l, LRN.Size);
            var alpha = _context.GetArgumentConstScalar<float>(l, LRN.Alpha);
            var beta = _context.GetArgumentConstScalar<float>(l, LRN.Beta);
            var k = _context.GetArgumentConstScalar<float>(l, LRN.Bias);
            return torch.nn.LocalResponseNorm(size, alpha, beta, k).forward(input);
        }
    }
}