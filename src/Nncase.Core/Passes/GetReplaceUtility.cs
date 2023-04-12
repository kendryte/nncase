// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reflection;
using NetFabric.Hyperlinq;
using Nncase.IR;
using static Nncase.IR.F.Tensors;
using Fx = System.Func<Nncase.IR.Expr, Nncase.IR.Expr>;
using ParameterInfo = Nncase.IR.ParameterInfo;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Passes;

public static class Utility
{
    /// <summary>
    /// apply a func with preprocess input.
    /// </summary>
    public static Fx Apply(Func<Fx, Fx> func, Fx inputCtor)
    {
        return input =>
        {
            if (input.CheckedType is null)
            {
                input.InferenceType();
            }

            if (input is Tuple inputs)
            {
                // todo: process tuple
                // need tuple param: Concat Stack
                // when input is tuple, Output is always expr that not tuple
                throw new InvalidOperationException("Not supported when input is a tuple");
            }

            return func(inputCtor)(input);
        };
    }

    /// <summary>
    /// insert Cast before and after call
    /// Cast(inputCtor(Cast(input, BF16)), input.Datatype)
    /// return a call like function that take a input and return a call.
    /// </summary>
    /// <param name="inputCtor"> a function take a input. </param>
    /// <returns> return a call like function that only take a input args. </returns>
    public static Fx WithTmpBF16(Fx inputCtor)
    {
        Fx WithTmpBF16Impl(Fx inCtor) => input => Cast(
            inCtor(Cast(input, DataTypes.BFloat16)),
            input.CheckedDataType);

        return Apply(WithTmpBF16Impl, inputCtor);
    }

    public static Fx WithTmpType(Fx inputCtor, DataType dt)
    {
        Fx WithTmpTypeImpl(Fx inputCtor) => input =>
            Cast(inputCtor(Cast(input, dt)), input.CheckedDataType);

        return Apply(WithTmpTypeImpl, inputCtor);
    }

    public static Fx WithTmp4DShape(Fx inputCtor, int[] originOutShape)
    {
        Fx WithTmpGNNEShape(Fx inCtor) =>
            input =>
                ((Func<int[], Expr>)(shape =>
                        Reshape(
                            inCtor(Reshape(input, Get4DGNNEShape(shape))),
                            originOutShape)))(input.CheckedShape.ToValueArray());

        return Apply(WithTmpGNNEShape, inputCtor);
    }

    internal static int[] Get4DGNNEShape(int[] dims)
    {
        if (dims.Length > 4)
        {
            throw new InvalidOperationException("dims Length should <= 4");
        }

        return Enumerable.Repeat(1, 4 - dims.Length).Concat(dims).ToArray();
    }
}
