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
public interface IPattern
{
    bool MatchLeaf(Expr expr);
}

/// <summary>
/// Pattern.
/// </summary>
/// <typeparam name="TResult">Match result type.</typeparam>
public interface IPattern<TResult> : IPattern
{
}
