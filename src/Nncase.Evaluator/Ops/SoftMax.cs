using Nncase.IR.NN;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitLogSoftMax(LogSoftMax logSoftMax)
        {
            var input = _context.GetArgument(logSoftMax, LogSoftMax.Input);
            var dim = _context.GetArgumentConst(logSoftMax, LogSoftMax.Axis).ToScalar<int>();
            return torchF.log_softmax(input, dim);
        }
        
        private torch.Tensor VisitSoftMax(SoftMax softMax)
        {
            var input = _context.GetArgument(softMax, SoftMax.Input);
            var dim = _context.GetArgumentConst(softMax, SoftMax.Axis).ToScalar<int>();
            return torchF.softmax(input, dim);
        }
        
        private torch.Tensor VisitSoftPlus(SoftPlus softPlus)
        {
            var input = _context.GetArgument(softPlus, SoftPlus.Input);
            return input.softplus();
        }
        
        // private torch.Tensor VisitSoftSign(SoftSign softSign)
        // {
        //     var input = _context.GetArgument(softSign, SoftSign.Input);
        //     torch.nn.
        // }
    }
}