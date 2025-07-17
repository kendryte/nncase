// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// GetPositionIds expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class GetPositionIds : Op
{
    /// <summary>
    /// Gets sequence length.
    /// </summary>
    public static readonly ParameterInfo SequenceLength = new(typeof(GetPositionIds), 0, "sequenceLength", IsDimensionType(), ParameterKind.Attribute);

    /// <summary>
    /// Gets kvCache.
    /// </summary>
    public static readonly ParameterInfo KVCache = new(typeof(GetPositionIds), 1, "kvCache", ParameterKind.Attribute);

    [Browsable(false)]
    public IRArray<SBP> NdSBP { get; }

    [Browsable(false)]
    public Placement Placement { get; }
}
