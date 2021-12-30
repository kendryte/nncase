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
            var input = _context.GetTorchArgument(celu, Celu.Input);
            return input.celu();
        }
        
        private torch.Tensor VisitElu(Elu elu)
        {
            var input = _context.GetTorchArgument(elu, Elu.Input);
            var alpha = _context.GetArgumentConst(elu, Elu.Alpha).ToScalar<double>();
            return torchF.elu(input, alpha);
        }

        private torch.Tensor VisitHardSwish(HardSwish hardSwish)
        {
            var input = _context.GetTorchArgument(hardSwish, HardSwish.Input);
            return input.hardswish();
        }

        private torch.Tensor VisitLeakyRelu(LeakyRelu leakyRelu)
        {
            var input = _context.GetTorchArgument(leakyRelu, LeakyRelu.Input);
            return input.leaky_relu(0.01);
        }
        
        private torch.Tensor VisitRelu(Relu relu)
        {
            var input = _context.GetTorchArgument(relu, Relu.Input);
            return input.relu();
        }
        
        private torch.Tensor VisitSelu(Selu selu)
        {
            var input = _context.GetTorchArgument(selu, Selu.Input);
            return input.selu();
        }
        
        private torch.Tensor VisitSigmoid(Sigmoid sigmoid)
        {
            var input = _context.GetTorchArgument(sigmoid, Sigmoid.Input);
            return input.sigmoid();
        }
    }
}