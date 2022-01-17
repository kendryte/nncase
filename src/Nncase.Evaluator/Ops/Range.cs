using TorchSharp;
using Nncase.IR;
using Range = Nncase.IR.Tensors.Range;
using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class RangeEvaluator : IEvaluator<Range>
    {
        public static Const Visit(EvaluatorContext context, Range range)
        {
            var begin = context.GetArgumentConstScalar<int>(range, Range.Begin);
            var end = context.GetArgumentConstScalar<int>(range, Range.End);
            var step = context.GetArgumentConstScalar<int>(range, Range.Step);
            return torch.arange(begin, end, step).ToConst();
        }
    }
}