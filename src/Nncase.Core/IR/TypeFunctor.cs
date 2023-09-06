// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Type functor.
/// </summary>
/// <typeparam name="TResult">Result visit type.</typeparam>
/// <typeparam name="TContext">Visit context.</typeparam>
public abstract class TypeFunctor<TResult, TContext>
{
    /// <summary>
    /// Visit type.
    /// </summary>
    /// <param name="type">Type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult VisitType(IRType type, TContext context)
    {
        return type switch
        {
            AnyType t => VisitType(t, context),
            NoneType t => VisitType(t, context),
            InvalidType t => VisitType(t, context),
            TensorType t => VisitType(t, context),
            TupleType t => VisitType(t, context),
            CallableType t => VisitType(t, context),
            DistTensorType t => VisitType(t, context),
            _ => DefaultVisitType(type, context),
        };
    }

    /// <summary>
    /// Visit any type.
    /// </summary>
    /// <param name="type">Any type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult VisitType(AnyType type, TContext context) => DefaultVisitType(type, context);

    /// <summary>
    /// Visit None type.
    /// </summary>
    /// <param name="type">None type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult VisitType(NoneType type, TContext context) => DefaultVisitType(type, context);

    /// <summary>
    /// Visit invalid type.
    /// </summary>
    /// <param name="type">Invalid type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult VisitType(InvalidType type, TContext context) => DefaultVisitType(type, context);

    /// <summary>
    /// Visit tensor type.
    /// </summary>
    /// <param name="type">Tensor type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult VisitType(TensorType type, TContext context) => DefaultVisitType(type, context);

    /// <summary>
    /// Visit pointer type.
    /// </summary>
    /// <param name="type">Pointer type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult VisitType(PointerType type, TContext context) => DefaultVisitType(type, context);

    /// <summary>
    /// Visit tuple type.
    /// </summary>
    /// <param name="type">Tuple type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult VisitType(TupleType type, TContext context) => DefaultVisitType(type, context);

    /// <summary>
    /// Visit callable type.
    /// </summary>
    /// <param name="type">Callable type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult VisitType(CallableType type, TContext context) => DefaultVisitType(type, context);

    /// <summary>
    /// Visit dist tensor type.
    /// </summary>
    /// <param name="type">dist tensor type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult VisitType(DistTensorType type, TContext context) => DefaultVisitType(type, context);

    /// <summary>
    /// Default visit routine.
    /// </summary>
    /// <param name="type">Type.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    public virtual TResult DefaultVisitType(IRType type, TContext context)
    {
        throw new NotImplementedException($"Unhandled visit routine for {type.GetType()}.");
    }
}
