using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using DotBuilder.Attributes;
using NnCase.IR.Serialization;

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

        public T AddNode<T>(T node)
            where T : Node
        {
            _nodes.Add(node);

            if (node is InputNode i)
                _inputs.Add(i);
            else if (node is OutputNode o)
                _outputs.Add(o);
            return node;
        }

        public void Collect()
        {
            var reachableNodes = new HashSet<Node>();
            var visitor = new RelayDfsVisitor(n => reachableNodes.Add(n));

            visitor.Visit(this);

            _nodes.RemoveAll(x =>
            {
                if (!reachableNodes.Contains(x))
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

        public void AssignNames()
        {
            var names = new HashSet<string>();
            var visitor = new RelayDfsVisitor(n =>
            {
                int i = 0;
                while (string.IsNullOrEmpty(n.Name) || names.Contains(n.Name))
                    n.Name = $"{n.GetType().Name}_{i++}";
                names.Add(n.Name);
            });
            visitor.Visit(this);
        }

        public void DumpDotGraph(Stream output)
        {
            AssignNames();
            var nodeDumps = new Dictionary<string, DumpContext>();
            _nodes.ForEach(n =>
            {
                var context = new DumpContext();
                n.Dump(context);
                nodeDumps.Add(n.Name, context);
            });
            var edges = new List<(string from, string to, Shape shape)>();
            var edgeVisitor = new RelayDfsVisitor(n =>
            {
                foreach (var output in n.Outputs)
                {
                    foreach (var input in output.Connections)
                    {
                        edges.Add((n.Name, input.Owner.Name, output.Shape));
                    }
                }
            });
            edgeVisitor.Visit(this);

            var graph = DotBuilder.Statements.Graph.Directed("graph")
                .WithNodeAttributesOf(
                    DotBuilder.Attributes.Shape.Record)
                .Containing(from n in nodeDumps
                            let records = new[] { n.Value.Title }.Concat(n.Value.Attributes.Select(x => x.Key + " " + x.Value))
                            select DotBuilder.Statements.Node.Name(n.Key)
                            .WithAttributesOf(Label.Set($"{{{string.Join('|', records)}}}")))
                .Containing(from e in edges
                            let label = string.Join('x', e.shape.ToArray())
                            select DotBuilder.Statements.Edge.Between(e.@from, e.to)
                            .WithAttributesOf(Label.Set(label)));
            using (var sw = new StreamWriter(output, leaveOpen: true))
                sw.Write(graph.Render());
        }
    }
}
