// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using OrtKISharp;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using Pad = Nncase.IR.NN.Pad;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Add preprocess in model.
/// </summary>
[RuleGenerator]
public sealed class AddPostProcess : ModulePass
{
    /// <summary>
    /// Main func for AddPreProcess.
    /// </summary>
    /// <param name="module"> The graph. </param>
    /// <param name="options"> RunPassContext. </param>
    /// <returns> Return a new graph with preprocess and postprocess. </returns>
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        var modelLayout = CompileSession.CompileOptions.ModelLayout;
        var outputLayout = CompileSession.CompileOptions.OutputLayout;

        var entry = (IR.Function)module.Entry!;

        if (modelLayout != outputLayout)
        {
            // Expr newOutput = entry.Body;
            var newOutput = outputLayout switch
            {
                "NHWC" when modelLayout == "NCHW" => Transpose(entry.Body, new[] { 0, 2, 3, 1 }),
                "NCHW" when modelLayout == "NHWC" => Transpose(entry.Body, new[] { 0, 3, 1, 2 }),
                _ => Transpose(
                    entry.Body,
                    Array.ConvertAll(
                        outputLayout.Replace(" ", string.Empty, StringComparison.OrdinalIgnoreCase).Split(","),
                        int.Parse)),
            };
            var newEntry = entry.With(body: newOutput);
            module.Remove(entry);
            module.Add(newEntry);
            module.Entry = newEntry;
        }

        return Task.FromResult(module);
    }
}
