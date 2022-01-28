// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

namespace Nncase.IR.Tensors
{
    public sealed record Stack() : Op
    {
        public static ParameterInfo Inputs = new(typeof(Stack), 0, "inputs");

        public static ParameterInfo Axis = new(typeof(Stack), 1, "axis", IsScalar() & IsIntegral());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TupleType inputs, TensorType axis)
        {
            if (context.GetArgument(this, Axis) is Const axis_con)
            {
                var axis_v = axis_con.ToScalar<int>();
                var ttypes = new TensorType[inputs.Count];
                foreach (var (i, input) in Enumerable.Range(0, inputs.Count).Zip(inputs))
                {
                    if (input is TensorType ttype)
                        ttypes[i] = ttype;
                    else
                        return new InvalidType("The Tuple Elements Type Must All Equals TensorType");
                }

                if (!ttypes.Skip(1).All(ttype => ttype.Shape == ttypes[0].Shape))
                    return new InvalidType("The Tuple Elements Shape Must All Equal!");

                if (ttypes[0].Shape.IsScalar)
                {
                    if (axis_v != 0)
                    {
                        return new InvalidType("Axis must be zero when stack scalar");
                    }

                    return ttypes[0] with { Shape = new Shape(inputs.Count) };
                }
                else
                {
                    var outshape = ttypes[0].Shape.ToList();
                    outshape.Insert(axis_v, inputs.Count);
                    return ttypes[0] with { Shape = new Shape(outshape) };
                }
            }

            return new InvalidType("The Stack Axis Must Be Const!");
        }
    }
}
