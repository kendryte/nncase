using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Runtime;

namespace Nncase.CodeGen.StackVM;

/// <summary>
/// StackVM runtime function.
/// </summary>
public class StackVMRTFunction : IRTFunction
{
    public IReadOnlyList<IRType> ParameterTypes => throw new NotImplementedException();

    public IRType ReturnType => throw new NotImplementedException();

    public ValueTask InitializeAsync()
    {
        throw new NotImplementedException();
    }

    public ValueTask InvokeAsync(IReadOnlyList<IValue> parameters, IValue ret)
    {
        throw new NotImplementedException();
    }

    public ValueTask UninitializeAsync()
    {
        throw new NotImplementedException();
    }
}
