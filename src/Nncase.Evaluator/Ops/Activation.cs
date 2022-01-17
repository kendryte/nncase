using Nncase.IR;
using Nncase.IR.NN;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public class CeluEvaluator : IEvaluator<Celu>
    {
        public static Const Visit(EvaluatorContext context, Celu celu)
        {
            var input = context.GetTorchArgument(celu, Celu.Input);
            return input.celu().ToConst();
        }
    }

    public class EluEvaluator : IEvaluator<Elu>
    {
        public static Const Visit(EvaluatorContext context, Elu elu)
        {
            var input = context.GetTorchArgument(elu, Elu.Input);
            var alpha = context.GetArgumentConst(elu, Elu.Alpha).ToScalar<double>();
            return torchF.elu(input, alpha).ToConst();
        }
    }

    public class HardSwishEvaluator : IEvaluator<HardSwish>
    {
        public static Const Visit(EvaluatorContext context, HardSwish hardSwish)
        {
            var input = context.GetTorchArgument(hardSwish, HardSwish.Input);
            return input.hardswish().ToConst();
        }
    }

    public class LeakyReluEvaluator : IEvaluator<LeakyRelu>
    {
        public static Const Visit(EvaluatorContext context, LeakyRelu leakyRelu)
        {
            var input = context.GetTorchArgument(leakyRelu, LeakyRelu.Input);
            return input.leaky_relu(0.01).ToConst();
        }
    }

    public class ReluEvaluator : IEvaluator<Relu>
    {
        public static Const Visit(EvaluatorContext context, Relu relu)
        {
            var input = context.GetTorchArgument(relu, Relu.Input);
            return input.relu().ToConst();
        }
    }

    public class SeluEvaluator : IEvaluator<Selu>
    {
        public static Const Visit(EvaluatorContext context, Selu selu)
        {
            var input = context.GetTorchArgument(selu, Selu.Input);
            return input.selu().ToConst();
        }
    }

    public class SigmoidEvaluator : IEvaluator<Sigmoid>
    {
        public static Const Visit(EvaluatorContext context, Sigmoid sigmoid)
        {
            var input = context.GetTorchArgument(sigmoid, Sigmoid.Input);
            return input.sigmoid().ToConst();
        }
    }
}