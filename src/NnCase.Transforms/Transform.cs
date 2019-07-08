using System;
using System.Collections.Generic;
using NnCase.IR;

namespace NnCase.Transforms
{
    public class TransformContext
    {
        public Graph Graph { get; }

        public List<Node> MatchedNodes { get; } = new List<Node>();

        public List<InputConnector> Inputs { get; } = new List<InputConnector>();

        public List<OutputConnector> Outputs { get; } = new List<OutputConnector>();

        public TransformContext(Graph graph)
        {
            Graph = graph;
        }
    }

    public abstract class Transform
    {
        protected virtual bool SkipSelfContainedCheck => false;

        public bool TryMatch(Node node, TransformContext context)
        {
            if (OnTryMatch(node, context))
            {
                if (!SkipSelfContainedCheck)
                {
                    // there exist input connectors out of the subgraph
                    foreach (var input in node.Inputs)
                    {
                        if (input.Connection != null)
                        {
                            if (!context.Inputs.Contains(input) && !context.MatchedNodes.Contains(input.Connection.Owner))
                                return false;
                        }
                    }

                    // there exist output connectors out of the subgraph
                    foreach (var output in node.Outputs)
                    {
                        foreach (var conn in output.Connections)
                        {
                            if (!context.Outputs.Contains(output) && !context.MatchedNodes.Contains(conn.Owner))
                                return false;
                        }
                    }
                }

                return true;
            }

            return false;
        }

        public abstract void Process(TransformContext context);

        protected abstract bool OnTryMatch(Node node, TransformContext context);

        public static void TransformGraph(Graph graph, IEnumerable<Transform> transforms)
        {
            bool nextPass = false;
            bool needRetry = false;
            Transform transform = null;

            var visitor = new RelayDfsVisitor(n =>
            {
                var context = new TransformContext(graph);
                if (transform.TryMatch(n, context))
                {
                    transform.Process(context);
                    needRetry = true;
                    return true;
                }

                return false;
            });

            do
            {
                nextPass = false;

                foreach (var srcTransform in transforms)
                {
                    transform = srcTransform;
                    needRetry = false;
                    visitor.Visit(graph);

                    if (needRetry)
                    {
                        nextPass = true;
                        graph.Collect();
                        break;
                    }
                }
            } while (nextPass);
        }
    }
}
