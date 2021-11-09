using System;
using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern;
using Nncase.Transform.Pattern.Math;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.Tensors;
using Nncase.Transform.Pattern.Tensors;
using System.Collections.Generic;

namespace Nncase.Transform.Rule
{
    public class SimplifyReduce : EGraphRule
    {
        private ReduceWrapper reduce;

        public SimplifyReduce()
        {
            Pattern = reduce = IsReduce((ReduceOp op) => true, IsWildCard(), IsConst(), IsConst(), IsConst());
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            reduce.Bind(result);

            var axis = reduce.Axis<Const>().ToTensor<int>();
            var keep_dims = reduce.KeepDims<Const>().ToScalar<bool>();
            if (axis.Length == 1 && !keep_dims)
            {
                var inshape = reduce.Input().CheckedShape;
                if (inshape.Rank > axis[0] + 1 && inshape[axis[0] + 1] == 1)
                {
                    var newshape = inshape.Take(axis[0]).ToList();
                    for (int i = axis[0] + 2; i < inshape.Rank; i++)
                    {
                        newshape.Add(inshape[i]);
                    }
                    return Reduce(reduce.ReduceOp,
                          Reshape(reduce.Input(), new Shape(newshape)),
                          reduce.Axis(),
                          reduce.InitValue(),
                          true);
                }
            }
            return null;
        }
    }
}