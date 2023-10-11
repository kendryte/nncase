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
/// VAE Merger for all.
/// </summary>
public sealed class VAEMerger : ExprCloner<Unit>
{
    private readonly IReadOnlyDictionary<Expr, Var> _multiVarMap;

    public VAEMerger(IReadOnlyDictionary<Expr, Var> multiVarMap)
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

/// <summary>
/// stable-disffusion VAE Decoder straight res-block.
/// </summary>
[RuleGenerator]
public sealed partial class FuseVAEDec1 : FusionMaker
{
    public override string ModuleKind { get; } = CPUTarget.Kind;

    public override Pattern Pattern => CreatePattern();

    private static Pattern CreatePattern()
    {
        var v1 = IsWildcard("input");
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

        // var v16 = IsConv2D(PadMode.Constant, v1, IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst(), IsTensorConst());
        var v17 = IsSwappableBinary(null!, "root", b => b.BinaryOp == BinaryOp.Add, v1, v15);

        return v17!;
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
        var merger = new VAEMerger(multiVarMap);
        var clonedRoot = merger.Clone(root, default);

        var callFusion = new Call(new Fusion("VAEDecResSraight", $"{nameof(FuseVAEDec1)}_{Count}", ModuleKind, clonedRoot, newInputs.OfType<Var>().ToArray()), input);
        return callFusion;
    }
}
