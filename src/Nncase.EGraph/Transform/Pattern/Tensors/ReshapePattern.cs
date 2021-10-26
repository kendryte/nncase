// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Transform.Pattern.Tensors
{
    /// <summary>
    /// Reshape expression.
    /// </summary>
    public record ReshapePattern() : OpPattern
    {
    }
}
