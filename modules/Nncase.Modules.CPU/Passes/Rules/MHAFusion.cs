// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
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
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules;

/// <summary>
/// MHA Merger for all.
/// </summary>
public sealed class MHAMerger : ExprCloner<Unit>
{
    private readonly IReadOnlyDictionary<Expr, Var> _multiVarMap;

    public MHAMerger(IReadOnlyDictionary<Expr, Var> multiVarMap)
    {
        _multiVarMap = multiVarMap;
    }

    protected override Expr VisitCall(Call expr, Unit context)
    {
        if (_multiVarMap.TryGetValue(expr, out var newVar))
        {
            return newVar;
        }

        return base.VisitCall(expr, context);
    }

    protected override Expr VisitLeafCall(Call expr, Unit context)
    {
        var target = Clone(expr.Target, context);
        var arguments = CloneArray(expr.Arguments, context);
        if (target is Binary)
        {
            arguments = arguments.Select(e => e switch { TensorConst { Value: Tensor { Shape.IsScalar: true } } tc => Const.FromTensor(Tensor.FromBytes(tc.ValueType.DType, tc.Value.BytesBuffer.ToArray(), new[] { 1 })), _ => e }).ToArray();
        }

        return expr.With(target: target, arguments: arguments);
    }

    protected override Expr VisitLeafVar(Var expr, Unit context)
    {
        if (_multiVarMap.TryGetValue(expr, out var newVar))
        {
            return newVar;
        }

        throw new InvalidOperationException();
    }
}

// pattern from BERT base
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
        var newInputs = new List<Expr>
        {
            new Var(x.CheckedType!),
            new Var(mask.CheckedType!),
        };

        var callFusion = new Call(new Fusion("MHABertBase", $"{nameof(FuseMHA1)}_{Count}", ModuleKind, root, newInputs.OfType<Var>().ToArray()), x, mask);
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
        var v3 = IsCall("v3", IsOp<LayerNorm>(), IsVArgs(v0, v1, v2));
        var v4 = IsTensorConst("v4");
        var v5 = IsCall("v5", IsOp<Unsqueeze>(), IsVArgs(v3, v4));
        var v6 = IsTensorConst("v6");
        var v7 = IsCall("v7", IsOp<MatMul>(), IsVArgs(v5, v6));
        var v8 = IsTensorConst("v8");
        var v9 = IsTensorConst("v9");
        var v10 = IsWildcard("position_ids");

        var v11 = IsCall("v11", IsOp<Gather>(_ => true), IsVArgs(v8, v10));
        var v12 = IsTensorConst("v12");
        var v13 = IsCall("v13", IsOp<Reshape>(), IsVArgs(v11, v12));
        var v14 = IsCall("v14", IsOp<Binary>(), IsVArgs(v7, v13));
        var v15 = IsTensorConst("v15");
        var v16 = IsTensorConst("v16");
        var v17 = IsTensorConst("v17");
        var v18 = IsTensorConst("v18");
        var v19 = IsCall("v19", IsOp<Slice>(), IsVArgs(v7, v15, v16, v17, v18));
        var v20 = IsCall("v20", IsOp<IR.Math.Unary>(), IsVArgs(v19));
        var v21 = IsTensorConst("v21");
        var v22 = IsCall("v22", IsOp<Slice>(), IsVArgs(v7, v21, v15, v17, v18));
        var v23 = IsTuple("v23", IsVArgs(v20, v22));

        var v24 = IsTensorConst("v24");
        var v25 = IsCall("v25", IsOp<Concat>(_ => true), IsVArgs(v23));
        var v26 = IsTensorConst("v26");
        var v27 = IsCall("v27", IsOp<Gather>(_ => true), IsVArgs(v26, v10));
        var v28 = IsCall("v28", IsOp<Reshape>(), IsVArgs(v27, v12));
        var v29 = IsCall("v29", IsOp<Binary>(), IsVArgs(v25, v28));
        var v30 = IsCall("v30", IsOp<Binary>(), IsVArgs(v14, v29));
        var v31 = IsTensorConst("v31");
        var v32 = IsCall("v32", IsOp<Unsqueeze>(), IsVArgs(v3, v31));
        var v33 = IsTensorConst("v33");
        var v34 = IsCall("v34", IsOp<MatMul>(), IsVArgs(v32, v33));
        var v35 = IsCall("v35", IsOp<Binary>(), IsVArgs(v34, v13));
        var v36 = IsCall("v36", IsOp<Slice>(), IsVArgs(v34, v15, v16, v17, v18));
        var v37 = IsCall("v37", IsOp<IR.Math.Unary>(), IsVArgs(v36));
        var v38 = IsCall("v38", IsOp<Slice>(), IsVArgs(v34, v21, v15, v17, v18));
        var v39 = IsTuple("v39", IsVArgs(v37, v38));

        var v40 = IsCall("v40", IsOp<Concat>(_ => true), IsVArgs(v39));
        var v41 = IsCall("v41", IsOp<Binary>(), IsVArgs(v40, v28));
        var v42 = IsCall("v42", IsOp<Binary>(), IsVArgs(v35, v41));
        var v43 = IsTensorConst("v43");
        var v44 = IsCall("v44", IsOp<Transpose>(), IsVArgs(v42, v43));
        var v45 = IsCall("v45", IsOp<MatMul>(), IsVArgs(v30, v44));
        var v46 = IsTensorConst("v46");
        var v47 = IsCall("v47", IsOp<Binary>(), IsVArgs(v45, v46));
        var v48 = IsWildcard("attn_mask");

        var v49 = IsCall("v49", IsOp<Binary>(), IsVArgs(v47, v48));
        var v50 = IsCall("v50", IsOp<Softmax>(), IsVArgs(v49, v24));
        var v51 = IsTensorConst("v51");
        var v52 = IsCall("v52", IsOp<Unsqueeze>(), IsVArgs(v3, v51));
        var v53 = IsTensorConst("v53");
        var v54 = IsCall("v54", IsOp<MatMul>(), IsVArgs(v52, v53));
        var v55 = IsCall("v55", IsOp<MatMul>(), IsVArgs(v50, v54));
        var v56 = IsTensorConst("v56");
        var v57 = IsCall("v57", IsOp<Transpose>(), IsVArgs(v55, v56));
        var v58 = IsTensorConst("v58");
        var v59 = IsCall("v59", IsOp<Reshape>(), IsVArgs(v57, v58));
        var v60 = IsTensorConst("v60");
        var v61 = IsCall("v61", IsOp<MatMul>(), IsVArgs(v59, v60));
        var v62 = IsCall("v62", IsOp<Binary>(), IsVArgs(v0, v61));
        var v63 = IsCall("v63", IsOp<LayerNorm>(), IsVArgs(v62, v1, v2));
        var v64 = IsTensorConst("v64");
        var v65 = IsCall("v65", IsOp<MatMul>(), IsVArgs(v63, v64));
        var v66 = IsCall("v66", IsOp<Swish>(), IsVArgs(v65));
        var v67 = IsTensorConst("v67");
        var v68 = IsCall("v68", IsOp<MatMul>(), IsVArgs(v63, v67));
        var v69 = IsCall("v69", IsOp<Binary>(), IsVArgs(v66, v68));
        var v70 = IsTensorConst("v70");
        var v71 = IsCall("v71", IsOp<MatMul>(), IsVArgs(v69, v70));
        var v72 = IsCall("root", IsOp<Binary>(), IsVArgs(v62, v71));

        return v72;
    }

    private Call? GetReplace(Call root, Expr hidden_in, Expr position_ids, Expr attn_mask)
    {
        var newInputs = new List<Expr>
        {
            new Var(hidden_in.CheckedType!),
            new Var(position_ids.CheckedType!),
            new Var(attn_mask.CheckedType!),
        };

        var multiVarMap = new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance)
        {
            { hidden_in, (Var)newInputs[0] },
            { position_ids, (Var)newInputs[1] },
            { attn_mask, (Var)newInputs[2] },
        };
        var merger = new MHAMerger(multiVarMap);
        var clonedRoot = merger.Clone(root, default);

        var callFusion = new Call(new Fusion("MHALLaMA65B", $"{nameof(FuseMHA2)}_{Count}", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), hidden_in, position_ids, attn_mask);
        return callFusion;
    }
}

/// <summary>
/// stable-disffusion text encoder.
/// </summary>
[RuleGenerator]
public sealed partial class FuseMHA3 : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var v19 = IsWildcard("input");
        var v20 = IsLayerNorm(2, 0.000009999999747378752f, true, v19, IsTensorConst(), IsTensorConst()); // f32[1,77,768]
        var v21 = IsMatMul(v20, IsTensorConst()); // f32[1,77,3072]
        var v22 = IsBinary(BinaryOp.Add, IsTensorConst(), v21); // f32[1,77,3072]
        var v23 = IsSwish(v22, IsTensorConst()); // f32[1,77,3072]
        var v24 = IsMatMul(v23, IsTensorConst()); // f32[1,77,768]
        var v25 = IsBinary(BinaryOp.Add, IsTensorConst(), v24); // f32[1,77,768]
        var v26 = IsBinary(BinaryOp.Add, v19, v25); // f32[1,77,768]
        var v27 = IsLayerNorm(2, 0.000009999999747378752f, true, v26, IsTensorConst(), IsTensorConst()); // f32[1,77,768]
        var v28 = IsMatMul(v27, IsTensorConst()); // f32[12,77,64]
        var v29 = IsBinary(BinaryOp.Add, v28, IsTensorConst()); // f32[12,77,64]
        var v30 = IsBinary(BinaryOp.Mul, v29, IsTensorConst()); // f32[12,77,64]
        var v31 = IsMatMul(v27, IsTensorConst()); // f32[12,77,64]
        var v32 = IsBinary(BinaryOp.Add, v31, IsTensorConst()); // f32[12,77,64]
        var v33 = IsTranspose(v32, IsTensorConst()); // f32[12,64,77]
        var v34 = IsMatMul(v30, v33); // f32[12,77,77]
        var v35 = IsBinary(BinaryOp.Add, v34, IsTensorConst()); // f32[12,77,77]
        var v36 = IsSoftmax(v35, IsTensorConst()); // f32[12,77,77]
        var v37 = IsMatMul(v27, IsTensorConst()); // f32[12,77,64]
        var v38 = IsBinary(BinaryOp.Add, v37, IsTensorConst()); // f32[12,77,64]
        var v39 = IsMatMul(v36, v38); // f32[12,77,64]
        var v40 = IsTranspose(v39, IsTensorConst()); // f32[77,12,64]
        var v41 = IsReshape(v40, IsTensorConst()); // f32[1,77,768]
        var v42 = IsMatMul(v41, IsTensorConst()); // f32[1,77,768]
        var v43 = IsBinary(BinaryOp.Add, IsTensorConst(), v42); // f32[1,77,768]
        var v44 = IsBinary(null, "root", BinaryOp.Add, v26, v43); // f32[1,77,768]
        return v44;
    }

    private Call? GetReplace(Call root, Expr input)
    {
        var newInputs = new List<Expr>
        {
            new Var(input.CheckedType!),
        };

        var multiVarMap = new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance)
        {
            { input, (Var)newInputs[0] },
        };
        var merger = new MHAMerger(multiVarMap);
        var clonedRoot = merger.Clone(root, default);

        var callFusion = new Call(new Fusion("MHASDTextEncoder", $"{nameof(FuseMHA3)}_{Count}", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), input);
        return callFusion;
    }
}
