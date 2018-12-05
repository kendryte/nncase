using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace NnCase.Converter.Model
{
    public abstract class Layer
    {
        private readonly List<InputConnector> _inputConnectors = new List<InputConnector>();
        private readonly List<OutputConnector> _outputConnectors = new List<OutputConnector>();

        public IReadOnlyList<InputConnector> InputConnectors => _inputConnectors;
        public IReadOnlyList<OutputConnector> OutputConnectors => _outputConnectors;

        public InputConnector AddInput(string name, ReadOnlySpan<int> dimensions)
        {
            var conn = new InputConnector(name, dimensions, this);
            _inputConnectors.Add(conn);
            return conn;
        }

        public OutputConnector AddOutput(string name, ReadOnlySpan<int> dimensions)
        {
            var conn = new OutputConnector(name, dimensions, this);
            _outputConnectors.Add(conn);
            return conn;
        }

        public void Plan(GraphPlanContext context)
        {
            if (!context.Planning.GetValueOrDefault(this))
            {
                context.Planning[this] = true;
                foreach (var input in _inputConnectors)
                    input.Connection?.From.Owner.Plan(context);

                OnPlanning(context);
            }
        }

        protected virtual void OnPlanning(GraphPlanContext context)
        {
        }
    }
}
