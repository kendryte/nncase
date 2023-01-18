// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// the interface that we can use parameterinfo the parameter.
/// </summary>
/// <typeparam name="T"></typeparam>
public interface IParameterList<T>
{
    /// <summary>
    /// get parameter info.
    /// </summary>
    /// <param name="parameter"></param>
    /// <returns></returns>
    public T this[ParameterInfo parameter] { get; }
}

/// <summary>
/// Call expression.
/// </summary>
public sealed record Call(Expr Target, IRArray<Expr> Parameters) : Expr, IParameterList<Expr>
{
    // /// <summary>
    // /// used by fake ir, represents that whether this op permit int 16 quant.
    // /// </summary>
    // public bool PermitInt16Quant = false;

    /// <summary>
    /// quant config with cosine, List of DataType represents data types for each input might be quantized, List of QuantParam represents quant params for each input.
    /// may be deleted in the future since there is EnodeBestQuantConfigWithCosine, reserve it now for debug and for unexpected usage when EnodeBestQuantConfigWithCosine is not enough.
    /// </summary>
    public List<Tuple<List<DataType>, List<List<QuantParam>>, float>> EnodeQuantConfigWithCosine;

    /// <summary>
    /// quant config with cosine, List of DataType represents data types for each input might be quantized, List of QuantParam represents quant params for each input.
    /// </summary>
    public Tuple<List<DataType>, List<List<QuantParam>>, float> EnodeBestQuantConfigWithCosine;

    /// <summary>
    /// adaround tmp output.
    /// </summary>
    public Tensor AdaRoundOutput;

    /// <summary>
    /// Initializes a new instance of the <see cref="Call"/> class.
    /// </summary>
    /// <param name="target">Call target.</param>
    /// <param name="parameters">Parameters.</param>
    public Call(Expr target, params Expr[] parameters)
        : this(target, new IRArray<Expr>(parameters.ToImmutableArray()))
    {
    }

    /// <summary>
    /// get param expr.
    /// </summary>
    /// <param name="parameter"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public Expr this[ParameterInfo parameter]
    {
        get
        {
            var type = Target.GetType();
            if (type == parameter.OwnerType)
            {
                return Parameters[parameter.Index];
            }
            else
            {
                throw new ArgumentOutOfRangeException($"Target {Target} doesn't have parameter: {parameter.Name}.");
            }
        }
    }

    public void ParametersForeach(Action<Expr, ParameterInfo> f)
    {
        var parameterInfos = ((Op)Target).Parameters.ToArray();
        for (int i = 0; i < Parameters.Count; i++)
        {
            f(Parameters[i], parameterInfos[i]);
        }
    }
}
