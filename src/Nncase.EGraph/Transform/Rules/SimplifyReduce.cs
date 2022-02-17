// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.Pattern;
using Nncase.Pattern.Math;
using static Nncase.Pattern.Utility;
using static Nncase.IR.F.Tensors;
using Nncase.Pattern.Tensors;
using System.Collections.Generic;

namespace Nncase.Transform.Rule
{
    public class SimplifyReduce : IRewriteRule
    {
        private ReduceWrapper reduce;

        public SimplifyReduce()
        {
            Pattern = reduce = IsReduce((ReduceOp op) => true, IsWildCard(), IsConst(), IsConst(), IsConst());
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            reduce.Bind(result);

            var axis = reduce.Axis<TensorConst>().Value.Cast<int>();
            var keep_dims = reduce.KeepDims<TensorConst>().Value.ToScalar<bool>();
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