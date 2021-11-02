// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Tensors;

namespace Nncase.Transform.Pattern.Tensors
{
    /// <summary>
    /// Cast expression.
    /// </summary>
    public record CastPattern(Func<Cast, bool> Cond) : OpPattern
    {
        public CastPattern(Func<DataType, bool> Cond) : this((Cast x) => Cond(x.NewType)) { }
    }
}
