// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Expression functor.
/// </summary>
/// <typeparam name="TPatternResult">Expression visit result type.</typeparam>
/// <typeparam name="TTypeResult">Type visit result type.</typeparam>
public abstract class PatternFunctor<TPatternResult, TTypeResult> : TypePatternFunctor<TTypeResult>
{
    /// <summary>
    /// Visit pattern.
    /// </summary>
    /// <param name="pattern">Expression.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(IPattern pattern)
    {
        return pattern switch
        {
            ExprPattern expr => Visit(expr),
            VarPattern var => Visit(var),
            TensorConstPattern con => Visit(con),
            TupleConstPattern con => Visit(con),
            ConstPattern con => Visit(con),
            FunctionPattern func => Visit(func),
            CallPattern call => Visit(call),
            TuplePattern tuple => Visit(tuple),
            IOpPattern op => Visit(op),
            _ => DefaultVisit(pattern),
        };
    }

    /// <summary>
    /// Visit expression pattern.
    /// </summary>
    /// <param name="pattern">Variable pattern.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(ExprPattern pattern) => DefaultVisit(pattern);

    /// <summary>
    /// Visit variable pattern.
    /// </summary>
    /// <param name="pattern">Variable pattern.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(VarPattern pattern) => DefaultVisit(pattern);

    /// <summary>
    /// Visit tensor constant pattern.
    /// </summary>
    /// <param name="pattern">Tensor constant pattern.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(TensorConstPattern pattern) => DefaultVisit(pattern);

    /// <summary>
    /// Visit tuple constant pattern.
    /// </summary>
    /// <param name="pattern">Tuple constant pattern.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(TupleConstPattern pattern) => DefaultVisit(pattern);

    /// <summary>
    /// Visit constant pattern.
    /// </summary>
    /// <param name="pattern">Constant pattern.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(ConstPattern pattern) => DefaultVisit(pattern);

    /// <summary>
    /// Visit function pattern.
    /// </summary>
    /// <param name="pattern">Variable pattern.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(FunctionPattern pattern) => DefaultVisit(pattern);

    /// <summary>
    /// Visit call pattern.
    /// </summary>
    /// <param name="pattern">Call pattern.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(CallPattern pattern) => DefaultVisit(pattern);

    /// <summary>
    /// Visit tuple pattern.
    /// </summary>
    /// <param name="pattern">Variable pattern.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(TuplePattern pattern) => DefaultVisit(pattern);

    /// <summary>
    /// Visit operator pattern.
    /// </summary>
    /// <param name="pattern">Operator pattern.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult Visit(IOpPattern pattern) => DefaultVisit(pattern);

    /// <summary>
    /// Default visit routine.
    /// </summary>
    /// <param name="pattern">Expression.</param>
    /// <returns>Result.</returns>
    public virtual TPatternResult DefaultVisit(IPattern pattern)
    {
        throw new NotImplementedException($"Unhandled visit routine for {pattern.GetType()}.");
    }
}
