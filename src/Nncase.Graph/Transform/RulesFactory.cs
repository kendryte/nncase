using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Pattern;
using Nncase.Pattern.Math;
using Nncase.IR.Math;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.F.NN;
using static Nncase.Pattern.Utility;
using Nncase.IR;
using Nncase.Evaluator;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;

namespace Nncase.Transform
{
    /// <inheritdoc/>
    internal sealed class ExprGeneratorVisitor : PatternVisitor<Expr, IRType>
    {

        private readonly IMatchResult _result;

        public ExprGeneratorVisitor(IMatchResult result) { _result = result; }

        /// <inheritdoc/>
        public override Expr VisitLeaf(CallPattern pattern)
        {
            return new Call(PatternMemo[pattern.Target], pattern.Parameters.Select(p => PatternMemo[p]).ToArray());
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(ConstPattern pattern)
        {
            return _result[pattern];
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(FunctionPattern pattern)
        {
            return new Function(PatternMemo[pattern.Body], pattern.Parameters.Select(p => PatternMemo[p]).ToArray());
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(OpPattern pattern) => pattern switch
        {
            BinaryPattern binary => new Binary(binary.BinaryOp!.Value),
            _ => throw new NotSupportedException($"Not Support Convert {pattern.GetType().Name}!")
        };

        /// <inheritdoc/>
        public override Expr VisitLeaf(TuplePattern pattern)
        {
            return new IR.Tuple(pattern.Fields.Select(f => PatternMemo[f]).ToArray());
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(VarPattern pattern)
        {
            return _result[pattern];
        }

        /// <inheritdoc/>
        public override Expr VisitLeaf(WildCardPattern pattern)
        {
            return _result[pattern];
        }
    }

    /// <summary>
    /// a template rule 
    /// </summary>
    public class TemplateRule : PatternRule
    {
        /// <summary>
        /// after expr
        /// </summary>
        public ExprPattern Rhs;

        /// <summary>
        /// predicate will be eval to bool
        /// </summary>
        public ExprPattern? Predicate;

        /// <summary>
        /// <see cref="RulesFactory.Rewrite(ExprPattern, ExprPattern, ExprPattern?)"/>
        /// </summary>
        public TemplateRule(ExprPattern lhs, ExprPattern rhs, ExprPattern? predicate = null)
        {
            Pattern = lhs;
            Rhs = rhs;
            Predicate = predicate;
        }

        /// <inheritdoc/>
        public override Expr? GetRePlace(IMatchResult result)
        {
            var converter = new ExprGeneratorVisitor(result);
            if (Predicate is null || (Predicate is not null && converter.Visit(Predicate).Eval().equal(torch.tensor(true)).ToBoolean()))
            {
                return converter.Visit(Rhs);
            }
            return null;
        }
    }

    /// <summary>
    /// Rules Factory 
    /// </summary>
    public static class RulesFactory
    {
        /// <summary>
        /// create the rewrite patternrule calss
        /// </summary>
        /// <param name="lhs">lhs pattern</param>
        /// <param name="rhs">rhs pattern expression</param>
        /// <param name="predicate"> predicate pattern expression </param>
        /// <returns> PatternRule </returns>
        public static PatternRule Rewrite(ExprPattern lhs, ExprPattern rhs, ExprPattern? predicate = null)
          => new TemplateRule(lhs, rhs, predicate);
    }
}