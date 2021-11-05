using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern;
using static Nncase.Transform.Pattern.F.Math;
using static Nncase.Transform.Pattern.F.Tensors;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using System.Numerics.Tensors;

namespace Nncase.Transform.Rule
{

    public class FoldSliceSlice : EGraphRule
    {
        public FoldSliceSlice()
        {
            WildCardPattern wcinput = "input";
            ConstPattern wcbegins1 = IsConstIntTensor(), wcends1 = IsConstIntTensor(), wcaxes1 = IsConstIntTensor(), wcstrides1 = IsConstIntTensor();
        }
    }

}