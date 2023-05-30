// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics.Metrics;
using System.IO;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Utilities;
using CallbacksRegister = System.Action<string, System.Action<Nncase.IR.Expr>>;
using TensorGetter = System.Func<Nncase.IR.Expr, Nncase.Tensor[]>;

namespace Nncase.Evaluator;

internal sealed class EvaluatorDumpManager : IDisposable
{
    private readonly IDumpper _dumpper;
    private readonly TensorGetter _tensorGetter;
    private readonly StreamWriter? _shapeWriter;

    private int _count;
    private bool _disposedValue;

    public EvaluatorDumpManager(IDumpper dumpper, TensorGetter tensorGetter)
    {
        _dumpper = dumpper;
        _tensorGetter = tensorGetter;

        if (_dumpper.IsEnabled(DumpFlags.Evaluator))
        {
            _shapeWriter = new StreamWriter(_dumpper.OpenFile("!out_shape_list"));
        }
    }

    public void RegisterDumpCallbacks(CallbacksRegister regBefore, CallbacksRegister regAfter)
    {
        if (_dumpper.IsEnabled(DumpFlags.Evaluator))
        {
            regBefore("DumpResult", expr => DumpCallArgs((Call)expr));
            regAfter("DumpResult", expr => DumpCall((Call)expr));
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private static string GetTargetName(Call call)
    {
        var target = DumpUtility.SnakeName(call.Target.GetType().Name);
        return target;
    }

    private void DumpCallArgs(Call call)
    {
        string target = GetTargetName(call);
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

    private void DumpCall(Call call)
    {
        string target = GetTargetName(call);

        // a bad tmp change
        var shape = !(call.CheckedType is TensorType) ? Shape.Scalar : call.CheckedShape;
        DumpCall(target, shape, sr =>
        {
            sr.WriteLine(target);
            var result = _tensorGetter(call);
            ValueDumper.DumpTensors(result, sr);
        });
    }

    private void UpdateOrder(string target, Shape shape)
    {
        _shapeWriter?.WriteLine($"{target}: {DumpUtility.SerializeShape(shape)}");
    }

    private void DumpCallParam(string target, ParameterInfo info, Action<StreamWriter> f)
    {
        using var sw = new StreamWriter(_dumpper.OpenFile($"{_count}${target}${info.Name}"));
        f(sw);
    }

    private void DumpCall(string target, Shape shape, Action<StreamWriter> f)
    {
        using var sw = new StreamWriter(_dumpper.OpenFile($"{_count}${target}"));
        f(sw);

        UpdateOrder(target, shape);
        ++_count;
    }

    private void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            if (disposing)
            {
                _shapeWriter?.Dispose();
            }

            _disposedValue = true;
        }
    }
}
