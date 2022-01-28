// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// Type inference context interface.
/// </summary>
public interface ITypeInferenceContext
{
    /// <summary>
    /// Get argument expression.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument expression.</returns>
    Expr GetArgument(Op op, ParameterInfo parameter);

    /// <summary>
    /// Get arguments expression.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="paramsInfo">ParamsInfo.</param>
    /// <returns>The arguments expression.</returns>
    Expr[] GetArguments(Op op, params ParameterInfo[] paramsInfo);

    /// <summary>
    /// Get argument type.
    /// </summary>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <returns>The argument type.</returns>
    IRType GetArgumentType(Op op, ParameterInfo parameter);
}

/// <summary>
/// Interrupt type inference.
/// </summary>
public sealed class TypeInferenceInterruptException : ApplicationException
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TypeInferenceInterruptException"/> class.
    /// </summary>
    /// <param name="reasonType">Reason type.</param>
    public TypeInferenceInterruptException(IRType reasonType)
        : this(reasonType, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TypeInferenceInterruptException"/> class.
    /// </summary>
    /// <param name="reasonType">Reason type.</param>
    /// <param name="innerException"></param>
    public TypeInferenceInterruptException(IRType reasonType, Exception? innerException)
        : base(null, innerException)
    {
        ReasonType = reasonType;
    }

    /// <summary>
    /// Gets reason type.
    /// </summary>
    public IRType ReasonType { get; }
}
