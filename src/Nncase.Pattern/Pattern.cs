// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Pattern;

/// <summary>
/// Pattern.
/// </summary>
public abstract partial record Pattern : IPattern
{
    /// <summary>
    /// Gets or sets hashcode cache, for speedup get hashcode.
    /// </summary>
    protected int? HashCode { get; set; }

    /// <summary>
    /// Gets pattern for CheckedType, defulat match IR Type.
    /// </summary>
    public TypePattern CheckedTypePattern { get; init; } = TypePatternUtility.IsIRType();

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode ??=
      System.HashCode.Combine(EqualityComparer<Type>.Default.GetHashCode(EqualityContract));

    /// <summary>
    /// Copy The **New** ExprPattern. NOTE the new pattern have different Id with old one, The there not equal.
    /// <remark> this copy not recursive </remark>
    /// </summary>
    /// <returns>Pattern.</returns>
    public abstract Pattern Copy();

    public virtual void Clear()
    {
    }

    public abstract bool MatchLeaf(Expr expr);

    /// <summary>
    /// Print members.
    /// </summary>
    /// <param name="builder">String builder.</param>
    /// <returns>Break print.</returns>
    protected virtual bool PrintMembers(StringBuilder builder)
    {
        builder.Append(this.DumpAsIL());
        return true;
    }

    /// <summary>
    /// Match The Expr Type.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Is Matched.</returns>
    private bool MatchCheckedType(Expr expr) => expr.CheckedType switch
    {
        IRType type => CheckedTypePattern.MatchLeaf(type),
        _ => false,
    };
}
