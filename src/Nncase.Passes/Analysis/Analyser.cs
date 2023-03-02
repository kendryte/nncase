// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;

namespace Nncase.Passes;

/// <summary>
/// the static mutator can create the mutator in the tir pass.
/// </summary>
public static class Analyser
{
    /// <summary>
    /// Analyis usedby informations.
    /// </summary>
    /// <param name="entry">Entry expressions.</param>
    /// <returns>result.</returns>
    public static IUsedByResult AnalysisUsedBy(Expr entry) => Analyses.UsedByAnalysisVisitor.Analysis(entry);
}
