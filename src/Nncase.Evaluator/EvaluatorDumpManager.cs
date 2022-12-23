// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.Utilities;
using CallbacksRegister = System.Action<string, System.Action<Nncase.IR.Expr>>;
using TensorGetter = System.Func<Nncase.IR.Expr, Nncase.Tensor[]>;

namespace Nncase.Evaluator;

public class EvaluatorDumpManager : DumpManager
{
    private readonly TensorGetter _tensorGetter;

    public EvaluatorDumpManager(TensorGetter tensorGetter)
    {
        _tensorGetter = tensorGetter;
    }

    public void DumpCallArgs(Call call)
    {
        var target = DumpUtility.SnakeName(call.Target.GetType().Name);
        var paramsInfo = ((Op)call.Target).Parameters.ToArray();

        call.ParametersForeach((param, paramInfo) =>
        {
            DumpCallParam(target, paramInfo, sr =>
            {
                var ps = _tensorGetter(param);
                ValueDumper.DumpTensors(ps, sr);
            });
        });
    }

    public void DumpCall(Call call, string root)
    {
        var target = call.Target.GetType().Name.ToLower();

        // a bad tmp change
        var shape = !(call.CheckedType is TensorType) ? Shape.Scalar : call.CheckedShape;
        DumpCall(target, shape, sr =>
        {
            sr.WriteLine(target);
            var result = _tensorGetter(call);
            ValueDumper.DumpTensors(result, sr);
        });
    }

    public void RegisterDumpCallbacks(CallbacksRegister regBefore, CallbacksRegister regAfter)
    {
        if (OpenDump)
        {
            regBefore("DumpResult", expr => DumpCallArgs((Call)expr));
            regAfter("DumpResult", expr => DumpCall((Call)expr, GetMaybeDumpDir()));
        }
    }
}
