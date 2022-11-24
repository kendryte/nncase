using System;
using System.Linq;
using Nncase.IR;
using Nncase.Utilities;
using CallbacksRegister = System.Action<string, System.Action<Nncase.IR.Expr>>;
using TensorGetter = System.Func<Nncase.IR.Expr, Nncase.Tensor[]>;

namespace Nncase.Evaluator;

public class EvaluatorDumpManager : DumpManager
{
    public EvaluatorDumpManager(TensorGetter tensorGetter)
    {
        TensorGetter = tensorGetter;
    }

    private TensorGetter TensorGetter;

    public void DumpCallArgs(Call call)
    {
        var target = DumpUtility.SnakeName(call.Target.GetType().Name);
        var paramsInfo = ((Op)call.Target).Parameters.ToArray();

        call.ParametersForeach((param, paramInfo) =>
        {
            DumpCallParam(target, paramInfo, sr =>
            {
                var ps = TensorGetter(param);
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
            var result = TensorGetter(call);
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