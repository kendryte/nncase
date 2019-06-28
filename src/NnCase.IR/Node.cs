using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR
{
    public abstract class Node
    {
        private readonly List<InputConnector> _inputs = new List<InputConnector>();
        private readonly List<OutputConnector> _outputs = new List<OutputConnector>();

        public IReadOnlyList<InputConnector> Inputs => _inputs;

        public IReadOnlyList<OutputConnector> Outputs => _outputs;

        protected InputConnector AddInput(string name, DataType type, Shape shape)
        {
            var conn = new InputConnector(name, this, type, shape);
            _inputs.Add(conn);
            return conn;
        }

        protected OutputConnector AddOutput(string name, DataType type, Shape shape)
        {
            var conn = new OutputConnector(name, this, type, shape);
            _outputs.Add(conn);
            return conn;
        }
    }
}
