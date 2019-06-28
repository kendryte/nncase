using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR
{
    public abstract class Visitor
    {
        private readonly HashSet<Node> _visited = new HashSet<Node>();

        public void Visit(Graph graph) =>
            Visit(graph.Outputs);

        public void Visit(IEnumerable<Node> roots)
        {
            _visited.Clear();

            foreach (var root in roots)
            {
                if (VisitStrategry(root))
                    break;
            }
        }

        protected virtual bool Visit(Node node) => false;

        protected abstract bool VisitStrategry(Node node);

        protected bool HasVisited(Node node) =>
            _visited.Contains(node);

        protected void MarkVisited(Node node) =>
            _visited.Add(node);
    }
}
