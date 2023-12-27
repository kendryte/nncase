// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.Passes.Rules.Neutral;
using Nncase.PatternMatch;
using Nncase.Targets;
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
            op => op is IR.Math.Unary or IR.Math.MatMul or IR.Tensors.Gather or IR.Math.Binary));

    private Call? GetReplace(Call call, Op op, IReadOnlyList<Expr> callParams)
    {
        var newInputs = new List<Expr>();
        for (int i = 0; i < callParams.Count; i++)
        {
            if (callParams[i] is Call or Var)
            {
                newInputs.Add(new Var(callParams[i].CheckedType!));
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
        var callFusion = new Call(new Fusion($"{op.GetType().Name}_{Count++}_kernel", ModuleKind, newCall, newInputs.OfType<Var>().ToArray()), newInputs.Select((e, i) => (e, i)).Where(p => p.e is Var).Select(p => callParams[p.i]).ToArray());
        return callFusion;
    }
}
