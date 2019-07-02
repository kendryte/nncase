using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation.Operators;
using NnCase.IR;

namespace NnCase.Evaluation
{
    public class EvaluatorRegistry
    {
        public static EvaluatorRegistry Default { get; } = CreateDefaultRegistry();

        private Dictionary<Type, Action<Node, Evaluator>> _evaluators;

        public EvaluatorRegistry()
        {
            _evaluators = new Dictionary<Type, Action<Node, Evaluator>>();
        }

        private EvaluatorRegistry(Dictionary<Type, Action<Node, Evaluator>> evaluators)
        {
            _evaluators = evaluators;
        }

        public void Add<T>(Action<T, Evaluator> evaluator)
            where T : Node
        {
            _evaluators[typeof(T)] = (n, e) => evaluator((T)n, e);
        }

        public bool TryInvoke(Node node, Evaluator evaluator)
        {
            if (_evaluators.TryGetValue(node.GetType(), out var e))
            {
                e(node, evaluator);
                return true;
            }

            return false;
        }

        public EvaluatorRegistry Clone()
        {
            return new EvaluatorRegistry(_evaluators);
        }

        private static EvaluatorRegistry CreateDefaultRegistry()
        {
            var registry = new EvaluatorRegistry();
            DefaultEvaulators.Register(registry);
            return registry;
        }
    }
}
