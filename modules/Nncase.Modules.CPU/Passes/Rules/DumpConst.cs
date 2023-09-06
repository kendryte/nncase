// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.SymbolStore;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules;

/// <summary>
/// Dump Weights of MM.
/// </summary>
[RuleGenerator]
public sealed partial class DumpMM : IRewriteRule
{
    public IPattern Pattern { get; } =
            IsMatMul(
                "mm",
                "mmCall",
                _ => true,
                IsWildcard("x"),
                IsTensorConst("w"));

    private Expr? GetReplace(Expr x, Call mmCall, TensorConst w)
    {
        var name = mmCall.Metadata.OutputNames?[0].Replace('/', '_') + "_weights.bin";
        var dir = DumpScope.Current.Directory;
        Directory.CreateDirectory(dir);
        using (var bw = new BinaryWriter(File.Open(Path.Join(dir, name), FileMode.Create)))
        {
            bw.Write(w.Value.BytesBuffer);
        }

        return null;
    }
}

/// <summary>
/// Dump Consts of layernorm.
/// </summary>
[RuleGenerator]
public sealed partial class DumpLayerNorm : IRewriteRule
{
    public IPattern Pattern { get; } =
            IsLayerNorm(
                "ln",
                "lnCall",
                _ => true,
                IsWildcard("x"),
                IsTensorConst("scale"),
                IsTensorConst("bias"));

    private Expr? GetReplace(Call lnCall, TensorConst scale, TensorConst bias)
    {
        var name = lnCall.Metadata.OutputNames?[0].Replace('/', '_');
        var dir = DumpScope.Current.Directory;
        Directory.CreateDirectory(dir);

        using (var bw = new BinaryWriter(File.Open(Path.Join(dir, name + "_scale.bin"), FileMode.Create)))
        {
            bw.Write(scale.Value.BytesBuffer);
        }

        using (var bw = new BinaryWriter(File.Open(Path.Join(dir, name + "_bias.bin"), FileMode.Create)))
        {
            bw.Write(bias.Value.BytesBuffer);
        }

        return null;
    }
}

/// <summary>
/// Dump Consts of Gather.
/// </summary>
[RuleGenerator]
public sealed partial class DumpGather : IRewriteRule
{
    public IPattern Pattern { get; } =
            IsGather(
                "gather",
                "gatherCall",
                _ => true,
                IsTensorConst("data"),
                IsWildcard("index"));

    private Expr? GetReplace(Call gatherCall, TensorConst data)
    {
        var name = gatherCall.Metadata.OutputNames?[0].Replace('/', '_') + "_data.bin";
        var dir = DumpScope.Current.Directory;
        Directory.CreateDirectory(dir);

        using (var bw = new BinaryWriter(File.Open(Path.Join(dir, name), FileMode.Create)))
        {
            bw.Write(data.Value.BytesBuffer);
        }

        return null;
    }
}
