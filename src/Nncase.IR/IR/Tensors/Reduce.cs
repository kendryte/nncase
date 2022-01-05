// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using static Nncase.IR.Utility;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// Reshape expression.
    /// </summary>
    public sealed record Reduce(ReduceOp ReduceOp) : Op
    {
        public static readonly ParameterInfo Input = new(typeof(Reduce), 0, "input");
        public static readonly ParameterInfo Axis = new(typeof(Reduce), 1, "axis", IsIntegral() & HasRank(1));
        public static readonly ParameterInfo InitValue = new(typeof(Reduce), 2, "initValue", IsScalar());
        public static readonly ParameterInfo KeepDims = new(typeof(Reduce), 3, "keepDims", IsScalar() & IsIntegral());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis,
          TensorType initValue, TensorType keepDims)
        {
            if (context.GetArgument(this, KeepDims) is Const keepDims_con &&
                context.GetArgument(this, Axis) is Const axis_con)
            {
                // todo:refactor
                var axes = axis_con.ToArray<int>();
                var outshape = input.Shape.ToValueArray();
                foreach (var a in axes)
                {
                    var ax = a < 0
                        ? a + input.Shape.Rank
                        : a;
                    if (keepDims_con.ToScalar<int>() == 1)
                        outshape[ax] = 1;
                    else
                    // todo: test
                        outshape[ax] = 0;
                }
                return input with { Shape = new Shape(outshape.Filter(x => x != -1)) };
            }
            return new InvalidType("Can't Infer Shape With Dynamic Input!");
        }
    }
}
