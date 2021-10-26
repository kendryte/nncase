// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform.Pattern.NN;


namespace Nncase.Transform.Pattern.F
{
    public static class NN
    {
        public static CallPattern Sigmoid(ExprPattern expr) => new CallPattern(new SigmoidPattern(x => true), expr);
    }
}
