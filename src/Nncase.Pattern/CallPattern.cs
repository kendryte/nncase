// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Pattern
{
    public sealed record CallPattern(ExprPattern Target, VArgsPattern Parameters) : ExprPattern
    {

        public CallPattern(Call call) : this((ExprPattern)call.Target, new FixedVArgsPattern(call.Parameters)) { }

        public bool MatchLeaf(Call call)
        {
            return MatchCheckedType(call);
        }

        public ExprPattern this[ParameterInfo parameter]
        {
            get => Parameters[parameter.Index];
        }


        public CallPattern(ExprPattern target, params ExprPattern[] parameters)
            : this(target, new FixedVArgsPattern(parameters))
        {
        }

    }

    public static partial class Utility
    {
        public static CallPattern IsCall(ExprPattern Target, VArgsPattern Parameters) => new CallPattern(Target, Parameters);

        public static CallPattern IsCall(ExprPattern Target, params ExprPattern[] Parameters) => new CallPattern(Target, Parameters);

    }
}