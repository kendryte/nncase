// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules;

// pattern from BERT
[RuleGenerator]
public sealed partial class FuseMHA1 : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var v0 = IsTensorConst("v0");
        var v1 = IsTensorConst("v1");
        var v2 = IsWildcard("x");

        var v3 = IsTensorConst("v3");
        var v4 = IsCall("v4", IsOp<MatMul>(), IsVArgs(v2, v3));
        var v5 = IsCall("v5", IsOp<Binary>(), IsVArgs(v1, v4));
        var v6 = IsTensorConst("v6");
        var v7 = IsCall("v7", IsOp<Reshape>(), IsVArgs(v5, v6));
        var v8 = IsTensorConst("v8");
        var v9 = IsCall("v9", IsOp<Transpose>(), IsVArgs(v7, v8));
        var v10 = IsTensorConst("v10");
        var v11 = IsTensorConst("v11");
        var v12 = IsCall("v12", IsOp<MatMul>(), IsVArgs(v2, v11));
        var v13 = IsCall("v13", IsOp<Binary>(), IsVArgs(v10, v12));
        var v14 = IsCall("v14", IsOp<Reshape>(), IsVArgs(v13, v6));
        var v15 = IsTensorConst("v15");
        var v16 = IsCall("v16", IsOp<Transpose>(), IsVArgs(v14, v15));
        var v17 = IsCall("v17", IsOp<MatMul>(), IsVArgs(v9, v16));
        var v18 = IsTensorConst("v18");
        var v19 = IsCall("v19", IsOp<Binary>(), IsVArgs(v17, v18));
        var v20 = IsWildcard("mask");

        var v21 = IsCall("v21", IsOp<Binary>(), IsVArgs(v19, v20));
        var v22 = IsTensorConst("v22");
        var v23 = IsCall("v23", IsOp<Reshape>(), IsVArgs(v21, v22));
        var v24 = IsTensorConst("v24");
        var v25 = IsCall("v25", IsOp<Softmax>(), IsVArgs(v23, v24));
        var v26 = IsTensorConst("v26");
        var v27 = IsCall("v27", IsOp<Reshape>(), IsVArgs(v25, v26));
        var v28 = IsTensorConst("v28");
        var v29 = IsTensorConst("v29");
        var v30 = IsCall("v30", IsOp<MatMul>(), IsVArgs(v2, v29));
        var v31 = IsCall("v31", IsOp<Binary>(), IsVArgs(v28, v30));
        var v32 = IsCall("v32", IsOp<Reshape>(), IsVArgs(v31, v6));
        var v33 = IsCall("v33", IsOp<Transpose>(), IsVArgs(v32, v8));
        var v34 = IsCall("v34", IsOp<MatMul>(), IsVArgs(v27, v33));
        var v35 = IsCall("v35", IsOp<Transpose>(), IsVArgs(v34, v8));
        var v36 = IsTensorConst("v36");
        var v37 = IsCall("v37", IsOp<Reshape>(), IsVArgs(v35, v36));
        var v38 = IsTensorConst("v38");
        var v39 = IsCall("v39", IsOp<MatMul>(), IsVArgs(v37, v38));
        var v40 = IsCall("v40", IsOp<Binary>(), IsVArgs(v0, v39));
        var v41 = IsCall("v41", IsOp<Binary>(), IsVArgs(v40, v2));
        var v42 = IsTensorConst("v42");
        var v43 = IsTensorConst("v43");
        var v44 = IsCall("ln", IsOp<LayerNorm>(), IsVArgs(v41, v42, v43));
        return v44;
    }

    private Call? GetReplace(Call ln, Expr x, Expr mask)
    {
        var newInputs = new List<Expr>();
        newInputs.Add(new Var(x.CheckedType!));
        newInputs.Add(new Var(mask.CheckedType!));

        var callFusion = new Call(new Fusion("MHA1", $"{ln.Target.GetType().Name}", "cpu", ln, newInputs.OfType<Var>().ToArray()), x, mask);
        return callFusion;
    }
}
