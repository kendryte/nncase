using Nncase.CodeGen;

namespace Nncase.IR;

public record Fusion(string Name, Expr Body, IRArray<Var> Parameters) : Nncase.IR.Function(Name,
    Body,
    Parameters)
{
    public ModuleType ModuleType;

    // a fusion should be a call
    public Fusion(string name, ModuleType moduleType, Expr body, params Var[] parameters) : this(name,
        (Call)body with { Attribute = CallAttr.Fusion }, parameters)
    {
        ModuleType = moduleType;
    }
}
