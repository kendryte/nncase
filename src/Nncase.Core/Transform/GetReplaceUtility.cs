using Nncase.IR;
using Tuple = Nncase.IR.Tuple;
using ParameterInfo = Nncase.IR.ParameterInfo;
using NetFabric.Hyperlinq;
using System.Reflection;
using static Nncase.IR.F.Tensors;

namespace Nncase.Transform;

using Fx = Func<Expr, Expr>;

public static class Utility
{
    /// <summary>
    /// apply a func with preprocess input
    /// </summary>
    /// <param name="func"></param>
    /// <param name="inputCtor"></param>
    /// <returns></returns>
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

    public static Func<Expr, Tuple> ApplyTuple(Fx inputCtor)
    {
        return input =>
        {
            if (input is Tuple inputs)
            {
                return new Tuple(inputs.Fields.Select(inputCtor));
            }
            else
            {
                throw new InvalidOperationException("Apply Tuple only support tuple input");
            }
        };
    }
    
    /// <summary>
    /// insert Cast before and after call
    /// Cast(inputCtor(Cast(input, BF16)), input.Datatype)
    /// return a call like function that take a input and return a call
    /// </summary>
    /// <param name="inputCtor"> a function take a input </param>
    /// <returns> return a call like function that only take a input args </returns>
    public static Fx withTmpBF16(Fx inputCtor)
    {
        Fx withTmpBF16Impl(Fx inCtor) => input => Cast(
            inCtor(Cast(input, DataTypes.BFloat16)),
            input.CheckedDataType);

        return Apply(withTmpBF16Impl, inputCtor);
    }
    
    public static Fx withTmpType(Fx inputCtor, DataType dt)
    {
        Fx withTmpTypeImpl(Fx inputCtor) => input =>
            Cast(inputCtor(Cast(input, dt)), input.CheckedDataType);

        return Apply(withTmpTypeImpl, inputCtor);
    }
    
    public static Fx withTmp4DShape(Fx inputCtor, int[] originOutShape)
    {
        Fx withTmpGNNEShape(Fx inCtor) =>
            (input =>
                ((Func<int[], Expr>)(shape =>
                        Reshape(
                            inCtor(Reshape(input, Get4DGNNEShape(shape))),
                            originOutShape)
                    ))(input.CheckedShape.ToValueArray()));

        return Apply(withTmpGNNEShape, inputCtor);
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