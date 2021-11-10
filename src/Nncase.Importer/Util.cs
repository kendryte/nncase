using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase
{
    public class Util
    {
        public static Call NHWCToNCHW(Expr input)
        {
            return F.Tensors.Transpose(input, new[] { 0, 3, 1, 2 });
        }
        
        public static Call NCHWToNHWC(Expr input)
        {
            return F.Tensors.Transpose(input, new[] { 0, 2, 3, 1 });
        }
    }
}