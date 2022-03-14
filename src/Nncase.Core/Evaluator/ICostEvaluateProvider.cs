using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;

namespace Nncase.Evaluator;

/// <summary>
/// Cost evaluate provider interface.
/// </summary>
public interface ICostEvaluateProvider
{
    /// <summary>
    /// Evaluate cost of the expression tree.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="varsValues">Optional vars' values.</param>
    /// <returns>Evaluate result.</returns>
    Cost EvaluateCost(Expr expr, IReadOnlyDictionary<Var, Cost>? varsValues = null);

    /// <summary>
    /// Evaluate cost of operator.
    /// </summary>
    /// <param name="op">Target operator.</param>
    /// <param name="context">Evaluate context.</param>
    /// <returns>Evaluate result.</returns>
    Cost EvaluateOpCost(Op op, ICostEvaluateContext context);
}
