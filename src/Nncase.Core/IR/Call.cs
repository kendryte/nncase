// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Utilities;

namespace Nncase.IR;

/// <summary>
/// the interface that we can use parameterinfo the parameter.
/// </summary>
public interface IParameterList<T>
{
    /// <summary>
    /// get parameter info.
    /// </summary>
    public T this[ParameterInfo parameter] { get; }
}

/// <summary>
/// Call expression.
/// </summary>
public sealed class Call : Expr, IParameterList<Expr>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Call"/> class.
    /// </summary>
    /// <param name="target">Call target.</param>
    /// <param name="arguments">Arguments.</param>
    public Call(Expr target, ReadOnlySpan<Expr> arguments)
        : base(ArrayUtility.Concat(target, arguments))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Call"/> class.
    /// </summary>
    /// <param name="target">Call target.</param>
    /// <param name="arguments">Arguments.</param>
    public Call(Expr target, params Expr[] arguments)
        : this(target, (ReadOnlySpan<Expr>)arguments)
    {
    }

    public Expr Target => Operands[0];

    public ReadOnlySpan<Expr> Arguments => Operands[1..];

    // /// <summary>
    // /// used by fake ir, represents that whether this op permit int 16 quant.
    // /// </summary>
    // public bool PermitInt16Quant = false;

    /// <summary>
    /// Gets or sets quant config with cosine, List of DataType represents data types for each input might be quantized, List of QuantParam represents quant params for each input.
    /// may be deleted in the future since there is EnodeBestQuantConfigWithCosine, reserve it now for debug and for unexpected usage when EnodeBestQuantConfigWithCosine is not enough.
    /// </summary>
    public List<Tuple<List<DataType>, List<List<QuantParam>>, float>>? EnodeQuantConfigWithCosine { get; set; }

    /// <summary>
    /// Gets or sets quant config with cosine, List of DataType represents data types for each input might be quantized, List of QuantParam represents quant params for each input.
    /// </summary>
    public Tuple<List<DataType>, List<List<QuantParam>>, float>? EnodeBestQuantConfigWithCosine { get; set; }

    /// <summary>
    /// get param expr.
    /// </summary>
    public Expr this[ParameterInfo parameter]
    {
        get
        {
            var type = Target.GetType();
            if (type == parameter.OwnerType)
            {
                return Arguments[parameter.Index];
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
        for (int i = 0; i < Arguments.Length; i++)
        {
            f(Arguments[i], parameterInfos[i]);
        }
    }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitCall(this, context);

    public Call With(Expr? target = null, Expr[]? arguments = null, IRMetadata? metadata = null)
    {
        var call = new Call(target ?? Target, arguments ?? Arguments);
        if (metadata != null && metadata!.OutputNames != null)
        {
            call.Metadata.OutputNames = metadata.OutputNames;
        }

        return call;
    }
}
