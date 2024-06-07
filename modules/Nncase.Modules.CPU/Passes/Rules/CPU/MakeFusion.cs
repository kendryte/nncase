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

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules;

[RuleGenerator]
internal sealed partial class CPUDeviceFusion : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => IsCallWildcard(
        "call",
        IsOp<Op>(
            "op",
            op => op is IR.Math.Unary /*or IR.Math.MatMul*/ or IR.Math.Binary));

    private Call? GetReplace(Call call, Op op, IReadOnlyList<Expr> callParams)
    {
        if (call.CheckedType is not DistributedType distributedType)
        {
            return null;
        }

        // note current not support.
        if (!Utilities.DistributedUtility.TryGetDividedTensorType(distributedType, out _))
        {
            return null;
        }

        var newInputs = new List<Expr>();
        for (int i = 0; i < callParams.Count; i++)
        {
            if (callParams[i] is Call or Var)
            {
                newInputs.Add(new Var(callParams[i].CheckedType!));
            }
            else
            {
                newInputs.Add(callParams[i]);
            }
        }

        var newCall = IR.F.CPU.Store(new Call(op, newInputs.Select(IR.F.CPU.Load).ToArray()));
        var callFusion = new Call(new Fusion($"{op.GetType().Name}_{Count++}_device", ModuleKind, newCall, newInputs.OfType<Var>().ToArray()), newInputs.Select((e, i) => (e, i)).Where(p => p.e is Var).Select(p => callParams[p.i]).ToArray());
        return callFusion;
    }
}

[RuleGenerator]
internal sealed partial class CPUSingleKernelFusion : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => IsCallWildcard(
        "call",
        IsOp<Op>(
            "op",
            op => op switch
            {
                IR.Math.Unary u => u.UnaryOp is UnaryOp.Abs or UnaryOp.Acos or UnaryOp.Acosh or UnaryOp.Asin or UnaryOp.Asinh or UnaryOp.Ceil or UnaryOp.Cos or UnaryOp.Cosh or UnaryOp.Exp or UnaryOp.Floor or UnaryOp.Log or UnaryOp.Neg or UnaryOp.Round or UnaryOp.Rsqrt or UnaryOp.Sign or UnaryOp.Sin or UnaryOp.Sinh or UnaryOp.Sqrt or UnaryOp.Square or UnaryOp.Tanh,
                IR.Math.MatMul => true,
                IR.Tensors.Gather => true,
                IR.Math.Binary b => b.BinaryOp is BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div or BinaryOp.Mod or BinaryOp.Min or BinaryOp.Max or BinaryOp.Pow,
                _ => false,
            })) with
    { TypePattern = TypePatternUtility.HasFixedShape() & TypePatternUtility.HasRank() };

    private Call? GetReplace(Call call, Op op, IReadOnlyList<Expr> callParams)
    {
        var newInputs = new List<Expr>();
        for (int i = 0; i < callParams.Count; i++)
        {
            if (callParams[i] is Call or Var or If or Marker)
            {
                newInputs.Add(new Var(callParams[i].CheckedType switch
                {
                    TensorType { IsScalar: true } t => t with { Shape = new Shape(1) },
                    var x => x,
                }));
            }
            else
            {
                if (callParams[i] is TensorConst { Value: Tensor { Shape.IsScalar: true } } tc)
                {
                    newInputs.Add(Const.FromTensor(Tensor.FromBytes(tc.CheckedDataType, tc.Value.BytesBuffer.ToArray(), new[] { 1 })));
                }
                else
                {
                    newInputs.Add(callParams[i]);
                }
            }
        }

        var newCall = new Call(op, newInputs.ToArray());
        var callFusion = new Call(new Fusion($"{op.GetType().Name}_{Count++}_kernel", ModuleKind, newCall, newInputs.OfType<Var>().ToArray()), newInputs.Select((e, i) => (e, i)).Where(p => p.e is Var).Select(p => callParams[p.i] switch
        {
            Expr { CheckedShape.IsScalar: true } e => IR.F.Tensors.Unsqueeze(e, new[] { 0 }),
            var e => e,
        }).ToArray());
        return callFusion;
    }
}

[RuleGenerator]
internal sealed partial class FuseMHA2 : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var v1 = IsWildcard("hidden_in");

        var v2 = IsTensorConst("v2");
        var v3 = IsTensorConst("v3");
        var v4 = IsCall("v4", IsOp<LayerNorm>(), IsVArgs(v1, v2, v3));
        var v5 = IsTensorConst("v5");
        var v6 = IsCall("v6", IsOp<Unsqueeze>(), IsVArgs(v4, v5));
        var v7 = IsTensorConst("v7");
        var v8 = IsCall("v8", IsOp<MatMul>(), IsVArgs(v6, v7));

        // var v9 = IsTensorConst("v9");
        var v10 = IsWildcard("left_gather");

        // var v11 = IsCall("v11", IsOp<Gather>(), IsVArgs(v9, v10));
        var v12 = IsTensorConst("v12");
        var v13 = IsCall("v13", IsOp<Reshape>(), IsVArgs(v10, v12));
        var v14 = IsCall("v14", IsOp<Binary>(), IsVArgs(v8, v13));
        var v15 = IsTensorConst("v15");
        var v16 = IsTensorConst("v16");
        var v17 = IsTensorConst("v17");
        var v18 = IsTensorConst("v18");
        var v19 = IsCall("v19", IsOp<Slice>(), IsVArgs(v8, v15, v16, v17, v18));
        var v20 = IsCall("v20", IsOp<Unary>(), IsVArgs(v19));
        var v21 = IsTensorConst("v21");
        var v22 = IsCall("v22", IsOp<Slice>(), IsVArgs(v8, v21, v15, v17, v18));
        var v23 = IsTuple("v23", IsVArgs(v20, v22));

        var v24 = IsCall("v24", IsOp<Concat>(), IsVArgs(v23));

        // var v25 = IsTensorConst("v25");
        // var v26 = IsCall("v26", IsOp<Gather>(), IsVArgs(v25, v10));
        var v26 = IsWildcard("right_gather");
        var v27 = IsCall("v27", IsOp<Reshape>(), IsVArgs(v26, v12));
        var v28 = IsCall("v28", IsOp<Binary>(), IsVArgs(v24, v27));
        var v29 = IsCall("v29", IsOp<Binary>(), IsVArgs(v14, v28));
        var v30 = IsTensorConst("v30");
        var v31 = IsCall("v31", IsOp<Unsqueeze>(), IsVArgs(v4, v30));
        var v32 = IsTensorConst("v32");
        var v33 = IsCall("v33", IsOp<MatMul>(), IsVArgs(v31, v32));
        var v34 = IsCall("v34", IsOp<Binary>(), IsVArgs(v33, v13));
        var v35 = IsCall("v35", IsOp<Slice>(), IsVArgs(v33, v15, v16, v17, v18));
        var v36 = IsCall("v36", IsOp<Unary>(), IsVArgs(v35));
        var v37 = IsCall("v37", IsOp<Slice>(), IsVArgs(v33, v21, v15, v17, v18));
        var v38 = IsTuple("v38", IsVArgs(v36, v37));

        var v39 = IsCall("v39", IsOp<Concat>(), IsVArgs(v38));
        var v40 = IsCall("v40", IsOp<Binary>(), IsVArgs(v39, v27));
        var v41 = IsCall("v41", IsOp<Binary>(), IsVArgs(v34, v40));
        var v42 = IsTensorConst("v42");
        var v43 = IsCall("v43", IsOp<Transpose>(), IsVArgs(v41, v42));
        var v44 = IsCall("v44", IsOp<MatMul>(), IsVArgs(v29, v43));
        var v45 = IsTensorConst("v45");
        var v46 = IsCall("v46", IsOp<Binary>(), IsVArgs(v44, v45));
        var v47 = IsWildcard("attn_mask");

        var v48 = IsCall("v48", IsOp<Binary>(), IsVArgs(v46, v47));
        var v49 = IsTensorConst("v49");
        var v50 = IsCall("v50", IsOp<Softmax>(), IsVArgs(v48, v49));
        var v51 = IsTensorConst("v51");
        var v52 = IsCall("v52", IsOp<Unsqueeze>(), IsVArgs(v4, v51));
        var v53 = IsTensorConst("v53");
        var v54 = IsCall("v54", IsOp<MatMul>(), IsVArgs(v52, v53));
        var v55 = IsCall("v55", IsOp<MatMul>(), IsVArgs(v50, v54));
        var v56 = IsTensorConst("v56");
        var v57 = IsCall("v57", IsOp<Transpose>(), IsVArgs(v55, v56));
        var v58 = IsTensorConst("v58");
        var v59 = IsCall("v59", IsOp<Reshape>(), IsVArgs(v57, v58));
        var v60 = IsTensorConst("v60");
        var v61 = IsCall("v61", IsOp<MatMul>(), IsVArgs(v59, v60));
        var v62 = IsCall("v62", IsOp<Binary>(), IsVArgs(v1, v61));
        var v2_ = IsTensorConst("v2_");
        var v3_ = IsTensorConst("v3_");
        var v63 = IsCall("v63", IsOp<LayerNorm>(), IsVArgs(v62, v2_, v3_));
        var v64 = IsTensorConst("v64");
        var v65 = IsCall("v65", IsOp<MatMul>(), IsVArgs(v63, v64));
        var v66 = IsTensorConst("v66");
        var v67 = IsCall("v67", IsOp<Swish>(), IsVArgs(v65, v66));
        var v68 = IsTensorConst("v68");
        var v69 = IsCall("v69", IsOp<MatMul>(), IsVArgs(v63, v68));
        var v70 = IsCall("v70", IsOp<Binary>(), IsVArgs(v67, v69));
        var v71 = IsTensorConst("v71");
        var v72 = IsCall("v72", IsOp<MatMul>(), IsVArgs(v70, v71));
        var v73 = IsCall("root", IsOp<Binary>(), IsVArgs(v62, v72));

        return v73;
    }

    private Call? GetReplace(Call root, Expr hidden_in, Expr left_gather, Expr right_gather, Expr attn_mask)
    {
        var newInputs = new List<Expr>
        {
            new Var(hidden_in.CheckedType!),
            new Var(left_gather.CheckedType!),
            new Var(right_gather.CheckedType!),
            new Var(attn_mask.CheckedType!),
        };

        var multiVarMap = new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance)
        {
            { hidden_in, (Var)newInputs[0] },
            { left_gather, (Var)newInputs[1] },
            { right_gather, (Var)newInputs[2] },
            { attn_mask, (Var)newInputs[3] },
        };
        var merger = new FusionMerger(multiVarMap);
        var clonedRoot = merger.Clone(root, default);

        var callFusion = new Call(new Fusion($"MHALLaMA65B_{nameof(FuseMHA2)}_{Count++}_kernel", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), hidden_in, left_gather, right_gather, attn_mask);
        return callFusion;
    }
}

/// <summary>
/// Convert QKV computation to MHA style.
/// %9 = MatMul(%2, const(f32[768,768]))
/// %10 = Add(BinaryOp.Add, const(f32[768]), %9)
/// %11 = Reshape(%10, const(i32[4] : {1,77,12,64}))
/// %12 = Transpose(%11, const(i64[4] : {0L,2L,1L,3L}))
/// %13 = Reshape(%12, const(i32[3] : {12,77,64})).
/// </summary>
[RuleGenerator]
internal sealed partial class CombineMHA : IRewriteRule
{
    public CombineMHA()
    {
        Pattern v0 = IsMatMul("mm", "mmCall", IsWildcard("x"), IsTensorConst("w"));

        var bias = IsAlt(
            IsBinary("add", "addCall", op => op.BinaryOp == BinaryOp.Add, IsTensorConst("bias"), v0),
            IsBinary("add", "addCall", op => op.BinaryOp == BinaryOp.Add, v0, IsTensorConst("bias")),
            v0);
        var scale = IsAlt(
            IsBinary("mul", "mulCall", op => op.BinaryOp == BinaryOp.Mul, bias, IsTensorConst("scale")),
            IsBinary("mul", "mulCall", op => op.BinaryOp == BinaryOp.Mul, IsTensorConst("scale"), bias),
            bias);

        var v1 = IsReshape("rshape", "rshapeCall", scale, IsTensorConst("newShape"));
        var v2 = IsTranspose("tp", "tpCall", v1, IsTensorConst("perm")) with { TypePattern = HasFixedShape() };
        Pattern = v2;
    }

    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr x, Call mmCall, TensorConst w, TensorConst newShape, int[] perm, IMatchResult matchResult)
    {
        var mmOutShape = mmCall.CheckedShape.ToValueArray();
        var wReshape = newShape.Value.ToArray<int>().TakeLast(2).ToArray();

        // TODO: add more patterns, only llama65b for now
        if (perm.Length == 4 && perm.SequenceEqual(new[] { 0, 2, 1, 3 })
             && wReshape.Aggregate(1, (x, y) => x * y) == mmOutShape[^1]
             && (mmOutShape.Length == 2 || (mmOutShape.Length == 3 && mmOutShape[0] == 1)))
        {
            var newW = IR.F.Tensors.Transpose(IR.F.Tensors.Reshape(w, new[] { -1, wReshape[0], wReshape[1] }), new[] { 1, 0, 2 });
            var newMm = IR.F.Tensors.MatMul(IR.F.Tensors.Unsqueeze(x, new[] { 1 }), newW);
            if (matchResult.GetValueOrDefault("bias") is TensorConst bias)
            {
                return null;
            }

            if (matchResult.GetValueOrDefault("scale") is TensorConst scale)
            {
                return null;
            }

            return newMm;
        }
        else if (perm.Length == 3 && perm.SequenceEqual(new[] { 1, 0, 2 })
            && wReshape.Aggregate(1, (x, y) => x * y) == mmOutShape[^1]
            && (mmOutShape.Length == 2 || (mmOutShape.Length == 3 && mmOutShape[0] == 1)))
        {
            var newW = IR.F.Tensors.Transpose(IR.F.Tensors.Reshape(w, new[] { -1, wReshape[0], wReshape[1] }), new[] { 1, 0, 2 });
            var newMm = IR.F.Tensors.MatMul(x, newW);
            if (matchResult.GetValueOrDefault("bias") is TensorConst bias)
            {
                newMm = IR.F.Math.Add(newMm, bias.Value.Shape.IsScalar ? bias : IR.F.Tensors.Reshape(bias, new[] { -1, 1, wReshape[1] }));
            }

            if (matchResult.GetValueOrDefault("scale") is TensorConst scale)
            {
                newMm = IR.F.Math.Mul(newMm, scale.Value.Shape.IsScalar ? scale : IR.F.Tensors.Reshape(scale, new[] { -1, 1, wReshape[1] }));
            }

            return newMm;
        }
        else if (perm.Length == 3 && perm.SequenceEqual(new[] { 1, 2, 0 })
            && wReshape.Aggregate(1, (x, y) => x * y) == mmOutShape[^1]
            && (mmOutShape.Length == 2 || (mmOutShape.Length == 3 && mmOutShape[0] == 1)))
        {
            var newW = IR.F.Tensors.Transpose(IR.F.Tensors.Reshape(w, new[] { -1, wReshape[0], wReshape[1] }), new[] { 1, 0, 2 });
            var newMm = IR.F.Tensors.MatMul(x, newW);
            if (matchResult.GetValueOrDefault("bias") is TensorConst bias)
            {
                newMm = IR.F.Math.Add(newMm, bias.Value.Shape.IsScalar ? bias : IR.F.Tensors.Reshape(bias, new[] { -1, 1, wReshape[1] }));
            }

            if (matchResult.GetValueOrDefault("scale") is TensorConst scale)
            {
                newMm = IR.F.Math.Mul(newMm, scale.Value.Shape.IsScalar ? scale : IR.F.Tensors.Reshape(scale, new[] { -1, 1, wReshape[1] }));
            }

            return IR.F.Tensors.Transpose(newMm, new[] { 0, 2, 1 });
        }

        return null;
    }
}
