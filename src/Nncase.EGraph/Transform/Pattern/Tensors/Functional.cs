// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform.Pattern.Tensors;
using static Nncase.Transform.Pattern.Utility;


namespace Nncase.Transform.Pattern.F
{
    public static class Tensor
    {
        public static CallPattern Transpose(ExprPattern input, ExprPattern perm) => Transpose(GetID(), input, perm);
        public static CallPattern Transpose(ID Id, ExprPattern input, ExprPattern perm) => new CallPattern(Id, new TransposePattern(x => true), input, perm);

        public static CallPattern Concat(TuplePattern input, ExprPattern axis) => Concat(GetID(), input, axis);
        public static CallPattern Concat(ID Id, TuplePattern input, ExprPattern axis) => new CallPattern(Id, new ConcatPattern(x => true), input, axis);
    }
}
