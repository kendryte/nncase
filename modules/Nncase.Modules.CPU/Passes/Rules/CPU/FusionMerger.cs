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
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules;

/// <summary>
/// Unet Merger for all.
/// </summary>
public sealed class FusionMerger : ExprCloner<Unit>
{
    private readonly IReadOnlyDictionary<Expr, Var> _multiVarMap;

    public FusionMerger(IReadOnlyDictionary<Expr, Var> multiVarMap)
    {
        _multiVarMap = multiVarMap;
    }

    protected override Expr VisitCall(Call expr, Unit context)
    {
        if (_multiVarMap.TryGetValue(expr, out var newVar))
        {
            if (expr.CheckedType is DistributedType d)
            {
                return IR.F.CPU.Boxing(newVar, d);
            }
            else
            {
                return newVar;
            }
        }

        return base.VisitCall(expr, context);
    }

    protected override Expr VisitLeafCall(Call expr, Unit context)
    {
        var target = Clone(expr.Target, context);
        var arguments = CloneArray(expr.Arguments, context);

        // if (target is Binary || target is Where)
        // {
        //     arguments = arguments.Select(e => e switch { TensorConst { Value: Tensor { Shape.IsScalar: true } } tc => Const.FromTensor(Tensor.FromBytes(tc.CheckedDataType, tc.Value.BytesBuffer.ToArray(), new[] { 1 })), _ => e }).ToArray();
        // }
        // if (target is Conv2D conv)
        // {
        //     var bias = (TensorConst)arguments[2];
        //     var fusedClamp = ((TensorConst)arguments[7]).Value.ToArray<float>();
        //     var newConv = IR.F.NN.Conv2D(arguments[0], arguments[1], Tensor.Zeros<float>(bias.CheckedShape), arguments[3], arguments[4], arguments[5], conv.PadMode, arguments[6], new[] { float.NegativeInfinity, float.PositiveInfinity });
        //     var newBias = IR.F.Math.Add(newConv, Tensor.FromBytes(bias.CheckedDataType, bias.Value.BytesBuffer.ToArray(), new[] { bias.CheckedShape[0].FixedValue, 1, 1 }));
        //     var newClamp = IR.F.Math.Clamp(newBias, fusedClamp[0], fusedClamp[1]);
        //     return newClamp;
        // }
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
