// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern.Math;
using Nncase.Transform.Pattern.Tensors;

namespace Nncase.Transform.Pattern
{
    public abstract record OpPattern() : ExprPattern
    {
        public bool MatchLeaf(Op op) => (this, op) switch
        {
            (BinaryPattern binaryPat, Binary binary) => binaryPat.MatchLeaf(binary),
            (ClampPattern clampPat, Clamp clamp) => clampPat.MatchLeaf(clamp),
            (UnaryPattern unaryPat, Unary unary) => unaryPat.MatchLeaf(unary),
            (TransposePattern transposePat, Transpose transpose) => transposePat.MatchLeaf(transpose),
            (SlicePattern slicePat, Slice slice) => slicePat.MatchLeaf(slice),
            (ConcatPattern concatPat, Concat concat) => concatPat.MatchLeaf(concat),
            (PadPattern padPat, Pad pad) => padPat.MatchLeaf(pad),
            (_, _) => false
        };
    }
}