using System;
using Nncase.IR;
using Nncase.IR.Tensors;
using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public class CastEvaluator : IEvaluator<Cast>
    {
        public Const Visit(EvaluatorContext context, Cast cast)
        {
            if (cast.NewType is PointerType)
            {
                var addr = context.GetArgumentConst(cast, Cast.Input);
                if ((addr.CheckedDataType is PrimType ptype) && (ptype.TypeCode != PrimTypeCode.UInt64))
                {
                    throw new InvalidCastException("Only The UInt64 Data Can't Be Cast To PointerType");
                }
                return addr with { ValueType = new(cast.NewType, addr.ValueType.Shape) };
            }
            var input = context.GetTorchArgument(cast, Cast.Input);
            return input.to_type(cast.NewType.ToTorchType()).ToConst();
        }
    }
}