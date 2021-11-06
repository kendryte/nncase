using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern.NN;
using Nncase.Transform.Pattern.Math;
using Nncase.Transform.Pattern.Tensors;
using static Nncase.Transform.Pattern.Utility;
using System.Numerics.Tensors;
using static Nncase.IR.F.Tensors;
using static Nncase.Transform.Pattern.F.NN;
using static Nncase.Transform.Pattern.F.Math;
using static Nncase.Transform.Pattern.F.Tensors;

namespace Nncase.Transform.Rule
{

    public class FuseClampConv2D : EGraphRule
    {
        //  cp;
        Conv2DWrapper conv2d;
        ClampWrapper cp;

        public FuseClampConv2D()
        {
            // conv2d = Conv2D();
            Pattern = cp = Clamp(IsWildCard(), IsConstIntSclar(), IsConstIntSclar());
        }

    }

}