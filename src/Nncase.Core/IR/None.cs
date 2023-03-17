// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace Nncase.IR;

/// <summary>
/// The None Expression is a placeholder for optional paramseter.
/// </summary>
public sealed class None : Expr, IEquatable<None?>
{
    private None()
        : base(Array.Empty<Expr>())
    {
    }

    /// <summary>
    /// Gets The default None expression instance.
    /// </summary>
    public static None Default => new();

    public static bool operator ==(None? left, None? right) => true;

    public static bool operator !=(None? left, None? right) => false;

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitNone(this, context);

    public None With() => Default;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as None);

    /// <inheritdoc/>
    public bool Equals(None? other) => other is not null;

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => 0;
}
