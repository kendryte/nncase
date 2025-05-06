// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Google.OrTools.ConstraintSolver;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.TIR;

namespace Nncase.Passes;

public sealed partial class NTTAffineSelectionPass : AffineSelectionPass
{
    private readonly CompileOptions _compileOptions;

    public NTTAffineSelectionPass(CompileOptions compileOptions)
        : base(NTTTarget.Kind)
    {
        _compileOptions = compileOptions;
    }

    protected override Expr SelectCall(Call call, Expr output)
    {
        switch (call.Target)
        {
            case IR.NTT.PackedBinary op:
                return SelectPackedBinary(op, call, output);
            case IR.NTT.PackedMatMul:
                return SelectMatMul((Op)call.Target, call, output);
            case IR.NTT.Pack op:
                return SelectPack(op, call, output);
            case IR.NTT.PackedReduce op:
                return SelectReduce(op, call, output);
            case IR.NTT.Unpack op:
                return SelectUnpack(op, call, output);
            case IR.Math.Binary op:
                return SelectBinary(op, call, output);
            case IR.Math.MatMul:
                return SelectMatMul((Op)call.Target, call, output);
            case IR.Math.Unary op:
                return SelectUnaryLike((Expr)call[IR.Math.Unary.Input], new TIR.NTT.Unary(op.UnaryOp), call, output);
            case IR.NN.Swish op:
                return SelectSwish(op, call, output);
            case IR.Tensors.Cast op:
                return SelectCast(op, call, output);
            case IR.Tensors.Transpose op:
                return SelectTranspose(op, call, output);
            default:
                return call;
        }
    }
}
