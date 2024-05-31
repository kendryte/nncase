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
/// Split expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Split : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(Split), 0, "input", ParameterKind.Input);

    /// <summary>
    /// Gets axis.
    /// </summary>
    public static readonly ParameterInfo Axis = new(typeof(Split), 1, "axis", IsScalar() & IsIntegral());

    /// <summary>
    /// Gets sections.
    /// </summary>
    public static readonly ParameterInfo Sections = new(typeof(Split), 2, "sections", IsIntegral() & HasRank(1));
}
