// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Tensors;

/// <summary>
/// ConstantOfShape expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class ConstantOfShape : Op
{
    /// <summary>
    /// Gets Shape.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(ConstantOfShape), 0, "shape", IsIntegral() & HasRank(1));

    /// <summary>
    /// Gets Value.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(ConstantOfShape), 1, "value");
}
