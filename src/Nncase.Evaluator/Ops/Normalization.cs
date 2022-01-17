using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public class BatchNormalizationEvaluator : IEvaluator<BatchNormalization>
    {
        public static Const Visit(EvaluatorContext context, BatchNormalization batchNorm)
        {
            var input = context.GetTorchArgument(batchNorm, BatchNormalization.Input);
            var eps = context.GetArgumentConst(batchNorm, BatchNormalization.Epsilon);
            var mom = context.GetArgumentConst(batchNorm, BatchNormalization.Momentum);
            var m = torch.nn.BatchNorm2d(input.shape[^3], eps.ToScalar<float>(), mom.ToScalar<float>());
            return m.forward(input).ToConst();
        }
    }

    public class InstanceNormalizationEvaluator : IEvaluator<InstanceNormalization>
    {
        public static Const Visit(EvaluatorContext context, InstanceNormalization i)
        {
            var input = context.GetTorchArgument(i, InstanceNormalization.Input);
            var eps = context.GetArgumentConst(i, InstanceNormalization.Epsilon).ToScalar<float>();
            var f = torch.nn.InstanceNorm2d(input.shape[1], eps);
            return f.forward(input).ToConst();
        }
    }

    public class LRNEvaluator : IEvaluator<LRN>
    {
        public static Const Visit(EvaluatorContext context, LRN l)
        {
            var input = context.GetTorchArgument(l, LRN.Input);
            var size = context.GetArgumentConstScalar<long>(l, LRN.Size);
            var alpha = context.GetArgumentConstScalar<float>(l, LRN.Alpha);
            var beta = context.GetArgumentConstScalar<float>(l, LRN.Beta);
            var k = context.GetArgumentConstScalar<float>(l, LRN.Bias);
            return torch.nn.LocalResponseNorm(size, alpha, beta, k).forward(input).ToConst();
        }
    }
}