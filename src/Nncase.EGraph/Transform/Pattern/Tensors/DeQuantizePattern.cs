// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.IR.Tensors;


namespace Nncase.Transform.Pattern.Tensors
{
    public sealed record DeQuantizePattern(Func<DeQuantize, bool> Cond) : OpPattern
    {
        public DeQuantizePattern(DeQuantize dequantize) : this(x => x == dequantize) { }

        public DeQuantizePattern(DataType targetType) : this(x => x.TargetType == targetType) { }

        public bool MatchLeaf(DeQuantize dequantize)
        {
            return Cond(dequantize) && MatchCheckedType(dequantize);
        }
    }
}