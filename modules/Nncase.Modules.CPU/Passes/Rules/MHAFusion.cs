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
        var v44 = IsCall("root", IsOp<LayerNorm>(), IsVArgs(v41, v42, v43));
        return v44;
    }

    private Call? GetReplace(Call root, Expr x, Expr mask)
    {
        var newInputs = new List<Expr>();
        newInputs.Add(new Var(x.CheckedType!));
        newInputs.Add(new Var(mask.CheckedType!));

        var callFusion = new Call(new Fusion("MHA1", $"{root.Target.GetType().Name}", "cpu", root, newInputs.OfType<Var>().ToArray()), x, mask);
        return callFusion;
    }
}

// pattern for LLaMA-65B
[RuleGenerator]
public sealed partial class FuseMHA2 : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var v0 = IsWildcard("hidden_in");

        var v1 = IsTensorConst("v1");
        var v2 = IsTensorConst("v2");
        var v3 = IsCall("v3", IsOp<LayerNorm>(), IsVArgs(v0,v1,v2));
        var v4 = IsTensorConst("v4");
        var v5 = IsCall("v5", IsOp<MatMul>(), IsVArgs(v3,v4));
        var v6 = IsTensorConst("v6");
        var v7 = IsCall("v7", IsOp<Reshape>(), IsVArgs(v5,v6));
        var v8 = IsTensorConst("v8");
        var v9 = IsCall("v9", IsOp<Transpose>(), IsVArgs(v7,v8));
        var v10 = IsTensorConst("v10");
        var v11 = IsTensorConst("v11");
        var v12 = IsWildcard("position_ids");

        var v13 = IsCall("v13", IsOp<Gather>(), IsVArgs(v10,v11,v12));
        var v14 = IsTensorConst("v14");
        var v15 = IsCall("v15", IsOp<Reshape>(), IsVArgs(v13,v14));
        var v16 = IsCall("v16", IsOp<Binary>(), IsVArgs(v9,v15));
        var v17 = IsTensorConst("v17");
        var v18 = IsTensorConst("v18");
        var v19 = IsTensorConst("v19");
        var v20 = IsTensorConst("v20");
        var v21 = IsCall("v21", IsOp<Slice>(), IsVArgs(v9,v17,v18,v19,v20));
        var v22 = IsCall("v22", IsOp<Unary>(), IsVArgs(v21));
        var v23 = IsTensorConst("v23");
        var v24 = IsCall("v24", IsOp<Slice>(), IsVArgs(v9,v23,v17,v19,v20));
        var v25 = IsTuple("v25", IsVArgs(v22,v24));

        var v26 = IsTensorConst("v26");
        var v27 = IsCall("v27", IsOp<Concat>(), IsVArgs(v25,v26));
        var v28 = IsTensorConst("v28");
        var v29 = IsCall("v29", IsOp<Gather>(), IsVArgs(v28,v11,v12));
        var v30 = IsCall("v30", IsOp<Reshape>(), IsVArgs(v29,v14));
        var v31 = IsCall("v31", IsOp<Binary>(), IsVArgs(v27,v30));
        var v32 = IsCall("v32", IsOp<Binary>(), IsVArgs(v16,v31));
        var v33 = IsTensorConst("v33");
        var v34 = IsCall("v34", IsOp<MatMul>(), IsVArgs(v3,v33));
        var v35 = IsCall("v35", IsOp<Reshape>(), IsVArgs(v34,v6));
        var v36 = IsCall("v36", IsOp<Transpose>(), IsVArgs(v35,v8));
        var v37 = IsCall("v37", IsOp<Binary>(), IsVArgs(v36,v15));
        var v38 = IsCall("v38", IsOp<Slice>(), IsVArgs(v36,v17,v18,v19,v20));
        var v39 = IsCall("v39", IsOp<Unary>(), IsVArgs(v38));
        var v40 = IsCall("v40", IsOp<Slice>(), IsVArgs(v36,v23,v17,v19,v20));
        var v41 = IsTuple("v41", IsVArgs(v39,v40));

        var v42 = IsCall("v42", IsOp<Concat>(), IsVArgs(v41,v26));
        var v43 = IsCall("v43", IsOp<Binary>(), IsVArgs(v42,v30));
        var v44 = IsCall("v44", IsOp<Binary>(), IsVArgs(v37,v43));
        var v45 = IsTensorConst("v45");
        var v46 = IsCall("v46", IsOp<Transpose>(), IsVArgs(v44,v45));
        var v47 = IsCall("v47", IsOp<MatMul>(), IsVArgs(v32,v46));
        var v48 = IsTensorConst("v48");
        var v49 = IsCall("v49", IsOp<Binary>(), IsVArgs(v47,v48));
        var v50 = IsWildcard("attn_mask");

        var v51 = IsCall("v51", IsOp<Binary>(), IsVArgs(v49,v50));
        var v52 = IsCall("v52", IsOp<Softmax>(), IsVArgs(v51,v26));
        var v53 = IsTensorConst("v53");
        var v54 = IsCall("v54", IsOp<MatMul>(), IsVArgs(v3,v53));
        var v55 = IsCall("v55", IsOp<Reshape>(), IsVArgs(v54,v6));
        var v56 = IsCall("v56", IsOp<Transpose>(), IsVArgs(v55,v8));
        var v57 = IsCall("v57", IsOp<MatMul>(), IsVArgs(v52,v56));
        var v58 = IsCall("v58", IsOp<Transpose>(), IsVArgs(v57,v8));
        var v59 = IsTensorConst("v59");
        var v60 = IsCall("v60", IsOp<Reshape>(), IsVArgs(v58,v59));
        var v61 = IsTensorConst("v61");
        var v62 = IsCall("v62", IsOp<MatMul>(), IsVArgs(v60,v61));
        var v63 = IsCall("v63", IsOp<Binary>(), IsVArgs(v0,v62));
        var v64 = IsCall("v64", IsOp<LayerNorm>(), IsVArgs(v63,v1,v2));
        var v65 = IsTensorConst("v65");
        var v66 = IsCall("v66", IsOp<MatMul>(), IsVArgs(v64,v65));
        var v67 = IsCall("v67", IsOp<Swish>(), IsVArgs(v66));
        var v68 = IsTensorConst("v68");
        var v69 = IsCall("v69", IsOp<MatMul>(), IsVArgs(v64,v68));
        var v70 = IsCall("v70", IsOp<Binary>(), IsVArgs(v67,v69));
        var v71 = IsTensorConst("v71");
        var v72 = IsCall("v72", IsOp<MatMul>(), IsVArgs(v70,v71));
        var v73 = IsCall("root", IsOp<Binary>(), IsVArgs(v63,v72));

        return v73;
    }

        private Call? GetReplace(Call root, Expr hidden_in, Expr position_ids, Expr attn_mask)
    {
        var newInputs = new List<Expr>();
        newInputs.Add(new Var(hidden_in.CheckedType!));
        newInputs.Add(new Var(position_ids.CheckedType!));
        newInputs.Add(new Var(attn_mask.CheckedType!));

        var callFusion = new Call(new Fusion("MHA2", $"{root.Target.GetType().Name}", "cpu", root, newInputs.OfType<Var>().ToArray()), hidden_in, position_ids, attn_mask);
        return callFusion;
    }
}


