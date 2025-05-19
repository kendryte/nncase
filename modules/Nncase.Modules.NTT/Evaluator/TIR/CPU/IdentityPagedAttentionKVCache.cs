﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DryIoc.ImTools;
using Google.OrTools.ConstraintSolver;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class IdentityPagedAttentionKVCacheEvaluator : ITypeInferencer<IdentityPagedAttentionKVCache>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, IdentityPagedAttentionKVCache target) => TupleType.Void;
}
