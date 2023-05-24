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
/// Fix Input Shape for TypeInfer
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class FixShape : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(FixShape), 0, "input");

    /// <summary>
    /// Gets Shape.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(FixShape), 1, "shape");
}
