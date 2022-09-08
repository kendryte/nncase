using Nncase.CodeGen;

namespace Nncase.IR;

/// <summary>
/// Fusion expression.
/// </summary>
public record Fusion(string Name, string ModuleKind, Expr Body, IRArray<Var> Parameters) : BaseFunction(Name, ModuleKind)
{
    private static int _globalFusionIndex = 0;

    /// <summary>
    /// Initializes a new instance of the <see cref="Fusion"/> class.
    /// </summary>
    /// <param name="module_kind"></param>
    /// <param name="parameters">Parameters.</param>
    /// <param name="body">Body.</param>
    public Fusion(string module_kind, Expr body, IRArray<Var> parameters)
        : this($"fusion_{_globalFusionIndex++}", module_kind, body, parameters)
    {
    }

    /// <summary>
    /// build function.
    /// </summary>
    /// <param name="module_kind"></param>
    /// <param name="body"></param>
    /// <param name="parameters"></param>
    public Fusion(string module_kind, Expr body, params Var[] parameters)
        : this($"fusion_{_globalFusionIndex++}", module_kind, body, new(parameters))
    {
    }

    /// <summary>
    /// get all parameter checked types.
    /// </summary>
    public override IEnumerable<IRType?> ParameterTypes => Parameters.Select(x => x.CheckedType);
}
