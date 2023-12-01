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

namespace Nncase.Passes.Rules.CPU;

[RuleGenerator]
internal sealed partial class CPUFusion : FusionMaker
{
    public override string ModuleKind { get; } = "cpu";

    public override Pattern Pattern => IsCallWildcard("call", IsOp<CPUKernelOp>("op"));

    private Call? GetReplace(Call call, CPUKernelOp op, IReadOnlyList<Expr> callParams)
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
                newInputs.Add(callParams[i]);
            }
        }

        var newCall = new Call(op, newInputs.ToArray());
        var callFusion = new Call(new Fusion($"{op.Target.GetType().Name}_{Count++}", ModuleKind, newCall, newInputs.OfType<Var>().ToArray()), newInputs.Select((e, i) => (e, i)).Where(p => p.e is Var).Select(p => callParams[p.i]).ToArray());
        return callFusion;
    }
}
