using System.Linq;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator
{
    public class Util
    {
        public static torch.Tensor ToTorchTensor(Const expr)
        {
            if (expr.ValueType.IsScalar)
            {
                return torch.tensor(expr.ToScalar<float>());
            }
            else
            {
                var shape = expr.CheckedShape.ToList().Select(x => x.FixedValue).ToList();
                return torch.tensor(expr.ToTensor<float>());
            }
        }

    }
}