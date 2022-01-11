using Nncase.IR.NN;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Operations;
using TorchSharp;
using static Tensorflow.Binding;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitLogSoftMax(LogSoftMax logSoftMax)
        {
            var input = _context.GetTorchArgument(logSoftMax, LogSoftMax.Input);
            var dim = _context.GetArgumentConst(logSoftMax, LogSoftMax.Axis).ToScalar<int>();
            return torchF.log_softmax(input, dim);
        }
        
        private torch.Tensor VisitSoftMax(SoftMax softMax)
        {
            var input = _context.GetTorchArgument(softMax, SoftMax.Input);
            var dim = _context.GetArgumentConst(softMax, SoftMax.Axis).ToScalar<int>();
            return torchF.softmax(input, dim);
        }
        
        private torch.Tensor VisitSoftPlus(SoftPlus softPlus)
        {
            var input = _context.GetTorchArgument(softPlus, SoftPlus.Input);
            return input.softplus();
        }
        
        private Tensorflow.Tensor VisitSoftSign(SoftSign softSign)
        {
            var input = _context.GetTFArgument(softSign, SoftSign.Input);
            // Tensorflow.Net no this interface
            return tf.Context.ExecuteOp("Softsign", null, new ExecuteOpArgs(input));
        }
    }
}