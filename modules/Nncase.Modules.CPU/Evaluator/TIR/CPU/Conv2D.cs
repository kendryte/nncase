// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.TIR.CPU;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.TIR.CPU;

public sealed class Conv2DEvaluator : ITypeInferencer<Conv2D>
{
    public IRType Visit(ITypeInferenceContext context, Conv2D target) => TupleType.Void;
}
