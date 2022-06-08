using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen;

public class LinkedFunction : ILinkedFunction
{
    public LinkedFunction(uint id, Callable sourceFunction, uint textBegin, uint textLength)
    {
        Id = id;
        switch (sourceFunction)
        {
            case Function func:
                ParameterTypes = func.Parameters.Select(x => x.TypeAnnotation).ToArray();
                ReturnType = func.Body.CheckedType ?? AnyType.Default;
                break;
            case TIR.PrimFunction pfunc:
                ParameterTypes = pfunc.Parameters.Select(x => x.ElemType).ToArray();
                ReturnType = pfunc.Body.CheckedType ?? AnyType.Default;
                break;
            default:
                throw new ArgumentOutOfRangeException();
        }
        TextBegin = textBegin;
        TextLength = textLength;
        Sections = sections;
    }

    public uint Id { get; }

    public IReadOnlyList<IRType> ParameterTypes { get; }

    public IRType ReturnType { get; }

    public uint TextBegin { get; }

    public uint TextLength { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
