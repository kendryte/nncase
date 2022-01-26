// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors
{
    public sealed record SpaceToBatch() : Op
    {
        public static readonly ParameterInfo Input = new(typeof(SpaceToBatch), 0, "input");

        public static readonly ParameterInfo BlockShape = new(typeof(SpaceToBatch), 1, "block_shape",
          IsRank(1) & IsIntegral());

        public static readonly ParameterInfo Paddings = new(typeof(SpaceToBatch), 2, "paddings",
          IsShape(new[] { Dimension.Unknown, 2 }) & IsIntegral());

        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType block_shape, TensorType paddings)
        {
            if (context.GetArgument(this, BlockShape) is Const block_shape_con &&
                 context.GetArgument(this, Paddings) is Const paddings_con)
            {
                var ts_block_shape = block_shape_con.ToTensor<int>();
                var ts_paddings = paddings_con.ToTensor<int>();
                int m = (int)ts_block_shape.Length;
                var padded_shape = input.Shape.ToList();
                for (int i = 1; i < 1 + m; i++)
                {
                    if (!padded_shape[i].IsUnknown)
                        padded_shape[i] += new Dimension(ts_paddings[i, 0] + ts_paddings[i, 1]);
                }
                var outshape = new List<Dimension> { padded_shape[0] };
                foreach (var i in Enumerable.Range(1, m))
                {
                    outshape.Add(padded_shape[i].IsUnknown ? Dimension.Unknown :
                                        ((padded_shape[i].FixedValue % ts_block_shape[i - 1] == 0) ?
                                          padded_shape[i].FixedValue / ts_block_shape[i - 1] :
                                          throw new TypeInferenceInterruptException(
                                            new InvalidType($"The Padded Shape Must Divides BlockShape!")
                                          )));
                }
                foreach (var i in Enumerable.Range(m + 1, outshape.Count - (m + 1)))
                {
                    outshape.Add(padded_shape[i]);
                }
                foreach (var block in ts_block_shape)
                {
                    outshape[0] = outshape[0].IsUnknown ? Dimension.Unknown : outshape[0].FixedValue * block;
                }
                return input with { Shape = new Shape(outshape) };
            }
            return new InvalidType("Can't Infer Shape With Dynamic Input!");
        }
    }
}
