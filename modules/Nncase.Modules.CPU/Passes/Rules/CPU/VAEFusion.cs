// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
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
/// stable-disffusion VAE Decoder res-block.
/// </summary>
[RuleGenerator]
public sealed partial class FuseVAEDecRes : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var vIn = IsWildcard("input");
        var vConv0 = IsConv2D(PadMode.Constant, vIn, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var vResize = IsResizeImage(ImageResizeMode.NearestNeighbor, ImageResizeTransformationMode.Asymmetric, ImageResizeNearestMode.Floor, false, vIn, IsWildcard(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var vConv1 = IsConv2D(PadMode.Constant, vConv0, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var vConv2 = IsConv2D(PadMode.Constant, vResize, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v1 = IsAlt(vConv2, vConv1, vIn);
        var v2 = IsReshape(v1, IsTensorConst());
        var v3 = IsInstanceNormalization(v2, IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v4 = IsReshape(v3, IsTensorConst());
        var v5 = IsBinary(BinaryOp.Mul, v4, IsTensorConst());
        var v6 = IsBinary(BinaryOp.Add, v5, IsTensorConst());
        var v7 = IsSwish(v6, IsTensorConst());
        var v8 = IsConv2D(PadMode.Constant, v7, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v9 = IsReshape(v8, IsTensorConst());
        var v10 = IsInstanceNormalization(v9, IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v11 = IsReshape(v10, IsTensorConst());
        var v12 = IsBinary(BinaryOp.Mul, v11, IsTensorConst());
        var v13 = IsBinary(BinaryOp.Add, v12, IsTensorConst());
        var v14 = IsSwish(v13, IsTensorConst());
        var v15 = IsConv2D(PadMode.Constant, v14, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v16 = IsConv2D(PadMode.Constant, v1, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v17 = IsSwappableBinary(null!, "root", b => b.BinaryOp == BinaryOp.Add, v16, v15);
        var v18 = IsSwappableBinary(null!, "root", b => b.BinaryOp == BinaryOp.Add, v1, v15);
        var root = IsAlt(v17, v18);
        return root;
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
        var callFusion = new Call(new Fusion($"VAEDecRes_{nameof(FuseVAEDecRes)}_{Count++}_kernel", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), input);
        return callFusion;
    }
}

/// <summary>
/// stable-disffusion VAE Decoder Head.
/// </summary>
[RuleGenerator]
public sealed partial class FuseVAEDecHead : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var v72 = IsWildcard("input");
        var v73 = IsReshape(v72, IsTensorConst());
        var v74 = IsInstanceNormalization(v73, IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v75 = IsReshape(v74, IsTensorConst());
        var v76 = IsBinary(BinaryOp.Mul, v75, IsTensorConst());
        var v77 = IsBinary(BinaryOp.Add, v76, IsTensorConst());
        var v78 = IsSwish(v77, IsTensorConst());
        var v79 = IsConv2D(null, "root", PadMode.Constant, v78, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        return v79!;
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
        var callFusion = new Call(new Fusion($"VAEDecHead_{nameof(FuseVAEDecHead)}_{Count++}_kernel", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), input);
        return callFusion;
    }
}

/// <summary>
/// stable-disffusion VAE Decoder MHA.
/// </summary>
[RuleGenerator]
public sealed partial class FuseVAEDecMHA : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var v0 = IsWildcard("input");
        var v1 = IsReshape(v0, IsTensorConst());
        var v2 = IsInstanceNormalization(v1, IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v3 = IsReshape(v2, IsTensorConst());
        var v4 = IsBinary(BinaryOp.Mul, v3, IsTensorConst());
        var v5 = IsBinary(BinaryOp.Add, v4, IsTensorConst());
        var v6 = IsReshape(v5, IsTensorConst());
        var v7 = IsTranspose(v6, IsTensorConst());
        var v8 = IsMatMul(v7, IsTensorConst());
        var v9 = IsBinary(BinaryOp.Add, IsTensorConst(), v8);
        var v10 = IsMatMul(v7, IsTensorConst());
        var v11 = IsBinary(BinaryOp.Add, IsTensorConst(), v10);
        var v12 = IsTranspose(v11, IsTensorConst());
        var v13 = IsMatMul(v9, v12);
        var v14 = IsBinary(BinaryOp.Mul, v13, IsTensorConst());

        // var v15 = IsBinary(BinaryOp.Add, v14, IsTensorConst());
        var v16 = IsSoftmax(v14, IsTensorConst());
        var v17 = IsMatMul(v7, IsTensorConst());
        var v18 = IsBinary(BinaryOp.Add, IsTensorConst(), v17);
        var v19 = IsMatMul(v16, v18);
        var v20 = IsMatMul(v19, IsTensorConst());
        var v21 = IsTranspose(v20, IsTensorConst());
        var v22 = IsBinary(BinaryOp.Add, IsTensorConst(), v21);
        var v23 = IsReshape(v22, IsTensorConst());
        var v24 = IsBinary(null!, "root", BinaryOp.Add, v23, v0);
        return v24!;
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
        var callFusion = new Call(new Fusion($"VAEDecMHA_{nameof(FuseVAEDecMHA)}_{Count++}_kernel", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), input);
        return callFusion;
    }
}
