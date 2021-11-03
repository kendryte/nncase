// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.IR.Tensors;


namespace Nncase.Transform.Pattern.Tensors
{
    public sealed record QuantizePattern(Func<Quantize, bool> Cond) : OpPattern
    {
        public QuantizePattern(Quantize quantize) : this(x => x == quantize) { }

        public QuantizePattern(DataType targetType) : this(x => x.TargetType == targetType) { }

        public bool MatchLeaf(Quantize quantize)
        {
            return Cond(quantize) && MatchCheckedType(quantize);
        }
    }
}