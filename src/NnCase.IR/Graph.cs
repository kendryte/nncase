using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR
{
    public class Graph
    {
        private readonly List<InputNode> _inputs = new List<InputNode>();
        private readonly List<OutputNode> _outputs = new List<OutputNode>();
        private readonly List<Node> _nodes = new List<Node>();

        public IReadOnlyList<InputNode> Inputs => _inputs;

        public IReadOnlyList<OutputNode> Outputs => _outputs;

        public IReadOnlyList<Node> Nodes => _nodes;

        public void AddNode(Node node)
        {
            _nodes.Add(node);

            if (node is InputNode i)
                _inputs.Add(i);
            else if (node is OutputNode o)
                _outputs.Add(o);
        }

        public void Collect()
        {
            var markVisitor = new MarkVisitor();
            markVisitor.Visit(this);

            _nodes.RemoveAll(x =>
            {
                if (!markVisitor.ReachableNodes.Contains(x))
                {
                    foreach (var input in x.Inputs)
                        input.ClearConnection();
                    foreach (var output in x.Outputs)
                        output.ClearConnections();
                    return true;
                }

                return false;
            });
        }

        private class MarkVisitor : DfsVisitor
        {
            public HashSet<Node> ReachableNodes { get; } = new HashSet<Node>();

            protected override bool Visit(Node node)
            {
                ReachableNodes.Add(node);
                return false;
            }
        }
    }
}
