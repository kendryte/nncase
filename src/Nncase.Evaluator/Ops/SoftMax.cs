using Nncase.IR.NN;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Operations;
using TorchSharp;
using Nncase.IR;
using static Tensorflow.Binding;
using Nncase.IR;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public class LogSoftMaxEvaluator : IEvaluator<LogSoftMax>
    {
        public static Const Visit(EvaluatorContext context, LogSoftMax logSoftMax)
        {
            var input = context.GetTorchArgument(logSoftMax, LogSoftMax.Input);
            var dim = context.GetArgumentConst(logSoftMax, LogSoftMax.Axis).ToScalar<int>();
            return torchF.log_softmax(input, dim).ToConst();
        }
    }

    public class SoftMaxEvaluator : IEvaluator<SoftMax>
    {
        public static Const Visit(EvaluatorContext context, SoftMax softMax)
        {
            var input = context.GetTorchArgument(softMax, SoftMax.Input);
            var dim = context.GetArgumentConst(softMax, SoftMax.Axis).ToScalar<int>();
            return torchF.softmax(input, dim).ToConst();
        }
    }

    public class SoftPlusEvaluator : IEvaluator<SoftPlus>
    {
        public static Const Visit(EvaluatorContext context, SoftPlus softPlus)
        {
            var input = context.GetTorchArgument(softPlus, SoftPlus.Input);
            return input.softplus().ToConst();
        }
    }
    public class SoftSignEvaluator : IEvaluator<SoftSign>
    {
    private Const Visit(EvaluatorContext context, SoftSign softSign)
        {
            var input = context.GetTFArgument(softSign, SoftSign.Input);
            // Tensorflow.Net no this interface
            return tf.Context.ExecuteOp("Softsign", null, new ExecuteOpArgs(input))[0].ToConst();
        }
    }
}