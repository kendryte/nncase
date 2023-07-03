// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
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
using Tuple = System.Tuple;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Add preprocess in model.
/// </summary>
[RuleGenerator]
public sealed class AddPostProcess : ModulePass
{
    /// <summary>
    /// Postprocess: support outputLayout.
    /// </summary>
    /// <param name="module"> The graph. </param>
    /// <param name="options"> RunPassContext. </param>
    /// <returns> Return a new graph with postprocess. </returns>
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        var preProcess = CompileSession.CompileOptions.PreProcess;
        var modelLayout = CompileSession.CompileOptions.ModelLayout;
        var outputLayout = CompileSession.CompileOptions.OutputLayout;

        var entry = (IR.Function)module.Entry!;

        if (preProcess && modelLayout != outputLayout && outputLayout != string.Empty)
        {
            Expr newOutput;
            if (entry.Body is IR.Tuple tuple)
            {
                newOutput = new IR.Tuple(tuple.Fields.ToArray().Select((field, i) => AddTranspose(outputLayout, modelLayout, field)).ToArray());
            }
            else
            {
                newOutput = AddTranspose(outputLayout, modelLayout, entry.Body);
            }

            var newEntry = entry.With(body: newOutput);
            module.Remove(entry);
            module.Add(newEntry);
            module.Entry = newEntry;
        }

        return Task.FromResult(module);
    }

    private static Expr AddTranspose(string outputLayout, string modelLayout, Expr entry)
    {
        var newOutput = outputLayout switch
        {
            "NHWC" when modelLayout == "NCHW" => Transpose(entry, new[] { 0, 2, 3, 1 }),
            "NCHW" when modelLayout == "NHWC" => Transpose(entry, new[] { 0, 3, 1, 2 }),
            _ => Transpose(
                entry,
                Array.ConvertAll(
                    outputLayout.Replace(" ", string.Empty, StringComparison.OrdinalIgnoreCase).Split(","),
                    int.Parse)),
        };
        return newOutput;
    }
}
