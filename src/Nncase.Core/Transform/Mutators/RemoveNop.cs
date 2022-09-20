using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Transform;
namespace Nncase.Transform.Mutators;

/// <summary>
/// remove the nop call from the body.
/// </summary>
internal sealed class RemoveNop : ExprMutator
{
    /// <inheritdoc/>
    /// shouldn't change the funciton
    public override Expr Visit(Function expr) => expr;

    public override Expr MutateLeaf(TIR.Sequential expr)
    {
        bool mutated = false;
        var bodys = new List<Expr>();
        foreach (var item in expr)
        {
            if (item is not Call { Target: TIR.Nop })
                bodys.Add(item);
            else
                mutated = true;
        }
        return mutated ? new TIR.Sequential(new IRArray<Expr>(bodys)) : expr;
    }
}
