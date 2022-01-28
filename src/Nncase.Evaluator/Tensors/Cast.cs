using Nncase.IR.Tensors;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class CastEvaluator : IEvaluator<Cast>
    {
        public Const Visit(EvaluatorContext context, Cast cast)
        {
            var input = context.GetTorchArgument(cast, Cast.Input);
            return input.to_type(cast.NewType.ToTorchType()).ToConst();
        }
    }
}