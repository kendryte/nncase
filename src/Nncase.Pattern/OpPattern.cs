// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Pattern.NN;
using Nncase.Pattern.Math;
using Nncase.Pattern.Tensors;
using System.Text;

namespace Nncase.Pattern
{
    public abstract partial record OpPattern : ExprPattern
    {
        public override int GetHashCode() => _hashcode ??=
          HashCode.Combine(
         EqualityComparer<Type>.Default.GetHashCode(EqualityContract),
         EqualityComparer<int>.Default.GetHashCode(Id));
    }
}