// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// GetPositionIds expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class GetPositionIds : Op
{
    /// <summary>
    /// Gets Q.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(GetPositionIds), 0, "input", ParameterKind.Attribute);

    /// <summary>
    /// Gets kvCache.
    /// </summary>
    public static readonly ParameterInfo KVCache = new(typeof(GetPositionIds), 1, "kvCache", ParameterKind.Attribute);
}
