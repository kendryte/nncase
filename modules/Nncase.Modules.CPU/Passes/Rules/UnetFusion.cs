// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Google.OrTools.LinearSolver;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules;

/// <summary>
/// stable-disffusion Unet spatial transformer.
/// </summary>
[RuleGenerator]
public sealed partial class FuseUnetSpatialTransformer : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        // resIn
        var v37 = IsWildcard("input");
        var v38 = IsReshape(v37, IsTensorConst());
        var v39 = IsInstanceNormalization(v38, IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v40 = IsReshape(v39, IsTensorConst());
        var v41 = IsBinary(BinaryOp.Mul, v40, IsTensorConst());
        var v42 = IsBinary(BinaryOp.Add, v41, IsTensorConst());
        var v43 = IsConv2D(PadMode.Constant, v42, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());

        // self attention
        var v44 = IsTranspose(v43, IsTensorConst());
        var v45 = IsReshape(v44, IsTensorConst());
        var v46 = IsLayerNorm(Axis: 2, Epsilon: 1E-05f, UseMean: true, v45, IsTensorConst(), IsTensorConst());
        var v47 = IsMatMul(v46, IsTensorConst());
        var v48 = IsReshape(v47, IsTensorConst());
        var v49 = IsTranspose(v48, IsTensorConst());
        var v50 = IsReshape(v49, IsTensorConst());
        var v51 = IsMatMul(v46, IsTensorConst());
        var v52 = IsReshape(v51, IsTensorConst());
        var v53 = IsTranspose(v52, IsTensorConst());
        var v54 = IsReshape(v53, IsTensorConst());
        var v55 = IsTranspose(v54, IsTensorConst());
        var v56 = IsMatMul(v50, v55);
        var v57 = IsBinary(BinaryOp.Mul, v56, IsTensorConst());
        var v58 = IsBinary(BinaryOp.Add, v57, IsTensorConst());
        var v59 = IsSoftmax(v58, IsTensorConst());
        var v60 = IsMatMul(v46, IsTensorConst());
        var v61 = IsReshape(v60, IsTensorConst());
        var v62 = IsTranspose(v61, IsTensorConst());
        var v63 = IsReshape(v62, IsTensorConst());
        var v64 = IsMatMul(v59, v63);
        var v65 = IsReshape(v64, IsTensorConst());
        var v66 = IsTranspose(v65, IsTensorConst());
        var v67 = IsReshape(v66, IsTensorConst());
        var v68 = IsMatMul(v67, IsTensorConst());
        var v69 = IsBinary(BinaryOp.Add, IsTensorConst(), v68);
        var v70 = IsBinary(BinaryOp.Add, v69, v45);

        // cross attention
        var v71 = IsLayerNorm(Axis: 2, Epsilon: 1E-05f, UseMean: true, v70, IsTensorConst(), IsTensorConst());
        var v72 = IsMatMul(v71, IsTensorConst());
        var v73 = IsReshape(v72, IsTensorConst());
        var v74 = IsTranspose(v73, IsTensorConst());
        var v75 = IsReshape(v74, IsTensorConst());
        var vencoderHiddenStates = IsWildcard("encoderHiddenStates");
        var v76 = IsMatMul(vencoderHiddenStates, IsTensorConst());
        var v77 = IsReshape(v76, IsTensorConst());
        var v78 = IsTranspose(v77, IsTensorConst());
        var v79 = IsReshape(v78, IsTensorConst());
        var v80 = IsTranspose(v79, IsTensorConst());
        var v81 = IsMatMul(v75, v80);
        var v82 = IsBinary(BinaryOp.Mul, v81, IsTensorConst());
        var v83 = IsBinary(BinaryOp.Add, v82, IsTensorConst());
        var v84 = IsSoftmax(v83, IsTensorConst());
        var v85 = IsMatMul(vencoderHiddenStates, IsTensorConst());
        var v86 = IsReshape(v85, IsTensorConst());
        var v87 = IsTranspose(v86, IsTensorConst());
        var v88 = IsReshape(v87, IsTensorConst());
        var v89 = IsMatMul(v84, v88);

        // FF
        var v90 = IsReshape(v89, IsTensorConst());
        var v91 = IsTranspose(v90, IsTensorConst());
        var v92 = IsReshape(v91, IsTensorConst());
        var v93 = IsMatMul(v92, IsTensorConst());
        var v94 = IsBinary(BinaryOp.Add, IsTensorConst(), v93);
        var v95 = IsBinary(BinaryOp.Add, v94, v70);
        var v96 = IsLayerNorm(Axis: 2, Epsilon: 1E-05f, UseMean: true, v95, IsTensorConst(), IsTensorConst());
        var v97 = IsMatMul(v96, IsTensorConst());
        var v98 = IsBinary(BinaryOp.Add, IsTensorConst(), v97);
        var v99 = IsSlice(v98, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v100 = IsSlice(v98, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v101 = IsGelu(v100, IsTensorConst());
        var v102 = IsBinary(BinaryOp.Mul, v99, v101);
        var v103 = IsMatMul(v102, IsTensorConst());
        var v104 = IsBinary(BinaryOp.Add, IsTensorConst(), v103);
        var v105 = IsBinary(BinaryOp.Add, v104, v95);
        var v106 = IsReshape(v105, IsTensorConst());
        var v107 = IsTranspose(v106, IsTensorConst());

        // res with result of ResBlock
        var v108 = IsConv2D(PadMode.Constant, v107, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var root = IsBinary(null, "root", BinaryOp.Add, v108, v37);

        return root!;
    }

    private Call? GetReplace(Call root, Expr input, Expr encoderHiddenStates)
    {
        var newInputs = new List<Expr>
        {
            new Var(input.CheckedType!),
            new Var(encoderHiddenStates.CheckedType!),
        };

        var multiVarMap = new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance)
        {
            { input, (Var)newInputs[0] },
            { encoderHiddenStates, (Var)newInputs[1] },
        };
        var merger = new FusionMerger(multiVarMap);
        var clonedRoot = merger.Clone(root, default);

        var callFusion = new Call(new Fusion("UnetSpatialTransformer", $"{nameof(FuseUnetSpatialTransformer)}_{Count++}", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), input, encoderHiddenStates);
        return callFusion;
    }
}

/// <summary>
/// stable-disffusion Unet res block.
/// </summary>
[RuleGenerator]
public sealed partial class FuseUnetResBlock : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        // 1st part with conv
        var vIn = IsWildcard("input");
        var vIn2 = IsWildcard("input2");
        var resizeIn = IsResizeImage(ImageResizeMode.NearestNeighbor, ImageResizeTransformationMode.Asymmetric, ImageResizeNearestMode.Floor, false, vIn, IsWildcard(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var convIn = IsConv2D(PadMode.Constant, IsAlt(resizeIn, vIn2), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var vtuple1 = IsTuple(IsVArgs(vIn, IsAlt(convIn, vIn2)));
        var vtuple2 = IsTuple(IsVArgs(convIn, vIn2));
        var vConcat = IsConcat(Axis: 1, IsAlt(vtuple2, vtuple1));
        var conv0 = IsConv2D(PadMode.Constant, vConcat, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var conv1 = IsConv2D(PadMode.Constant, vIn, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var conv2 = IsConv2D(PadMode.Constant, conv1, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v1 = IsReshape(IsAlt(vConcat, conv1, vIn), IsTensorConst());
        var v2 = IsInstanceNormalization(v1, IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v3 = IsReshape(v2, IsTensorConst());
        var v4 = IsBinary(BinaryOp.Mul, v3, IsTensorConst());
        var v5 = IsBinary(BinaryOp.Add, v4, IsTensorConst());
        var v6 = IsSwish(v5, IsTensorConst());
        var v7 = IsConv2D(PadMode.Constant, v6, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());

        // 2nd part with emb
        var v25 = IsWildcard("emb");
        var v26 = IsMatMul(v25, IsTensorConst());
        var v27 = IsBinary(BinaryOp.Add, v26, IsTensorConst());
        var v28 = IsReshape(v27, IsTensorConst());

        // 1st + 2nd
        var v29 = IsBinary(BinaryOp.Add, v7, v28);

        // res
        var v30 = IsReshape(v29, IsTensorConst());
        var v31 = IsInstanceNormalization(v30, IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v32 = IsReshape(v31, IsTensorConst());
        var v33 = IsBinary(BinaryOp.Mul, v32, IsTensorConst());
        var v34 = IsBinary(BinaryOp.Add, v33, IsTensorConst());
        var v35 = IsSwish(v34, IsTensorConst());
        var v36 = IsConv2D(PadMode.Constant, v35, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v37 = IsBinary(null, "root", BinaryOp.Add, conv0, v36);
        var v38 = IsBinary(null, "root", BinaryOp.Add, conv2, v36);
        var v39 = IsBinary(null, "root", BinaryOp.Add, conv1, v36);
        var v40 = IsBinary(null, "root", BinaryOp.Add, vIn, v36);
        var root = IsAlt(v37, v38, v39, v40);

        return root!;
    }

    private Call? GetReplace(Call root, IMatchResult result)
    {
        var oldInputs = new List<Expr> {
            (Expr)result["input"],
            (Expr)result["emb"],
        };
        if (result["input2"] is Expr expr)
        {
            oldInputs.Add(expr);
        }

        var newInputs = new List<Expr>();
        foreach (var i in oldInputs)
        {
            newInputs.Add(new Var(i.CheckedType!));
        }

        var multiVarMap = new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance);
        for (var i = 0; i < newInputs.Count; i++)
        {
            multiVarMap.Add(oldInputs[i], (Var)newInputs[i]);
        }

        var merger = new FusionMerger(multiVarMap);
        var clonedRoot = merger.Clone(root, default);

        var callFusion = new Call(new Fusion("UnetResBlock", $"{nameof(FuseUnetResBlock)}_{Count++}", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), oldInputs.ToArray());
        return callFusion;
    }
}

/// <summary>
/// stable-disffusion Unet TimeStep Embedding.
/// </summary>
[RuleGenerator]
public sealed partial class FuseUnetTimeEmb : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var vtimestep = IsWildcard("input");
        var v8 = IsExpand(vtimestep, IsTensorConst());
        var v9 = IsReshape(v8, IsTensorConst());
        var v10 = IsCast(DataTypes.Float32, CastMode.KDefault, v9);
        var v11 = IsBinary(BinaryOp.Mul, v10, IsTensorConst());
        var v12 = IsUnary(UnaryOp.Sin, v11);
        var v13 = IsUnary(UnaryOp.Cos, v11);
        var v14 = IsTuple(IsVArgs(v12, v13));
        var v15 = IsConcat(Axis: 1, v14);
        var v16 = IsSlice(v15, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v17 = IsSlice(v15, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v18 = IsTuple(IsVArgs(v16, v17));
        var v19 = IsConcat(Axis: 1, v18);
        var v20 = IsMatMul(v19, IsTensorConst());
        var v21 = IsBinary(BinaryOp.Add, v20, IsTensorConst());
        var v22 = IsSwish(v21, IsTensorConst());
        var v23 = IsMatMul(v22, IsTensorConst());
        var v24 = IsBinary(BinaryOp.Add, v23, IsTensorConst());
        var root = IsSwish(null, "root", v24, IsTensorConst());

        return root!;
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
        var merger = new FusionMerger(multiVarMap);
        var clonedRoot = merger.Clone(root, default);

        var callFusion = new Call(new Fusion("UnetTimeEmb", $"{nameof(FuseUnetTimeEmb)}_{Count++}", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), input);
        return callFusion;
    }
}
