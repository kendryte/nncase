// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffers;

/// <summary>
/// MatchBuffer op.
/// todo maybe need united matchbuffer and allocatebuffer
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class MatchBuffer : Op
{
    public static readonly ParameterInfo Input = new(typeof(MatchBuffer), 0, "input", IsTensor());

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
