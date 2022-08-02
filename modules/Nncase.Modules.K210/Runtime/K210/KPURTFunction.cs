using Nncase.IR;

namespace Nncase.Runtime.K210;

public class KPURTFunction
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