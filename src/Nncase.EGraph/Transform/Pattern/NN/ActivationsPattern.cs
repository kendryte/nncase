// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.NN;

namespace Nncase.Transform.Pattern.NN
{
    /// <summary>
    /// Sigmoid expression.
    /// </summary>
    public record SigmoidPattern(Func<Sigmoid, bool> Cond) : OpPattern
    {

    }
}
