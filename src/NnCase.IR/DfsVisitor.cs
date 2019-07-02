using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR
{
    public class DfsVisitor : Visitor
    {
        protected sealed override bool VisitStrategry(Node node)
        {
            if (!HasVisited(node))
            {
                MarkVisited(node);

                foreach (var input in node.Inputs)
                {
                    if (input.Connection != null)
                    {
                        if (VisitStrategry(input.Connection.Owner))
                            return true;
                    }
                }

                if (Visit(node))
                    return true;
            }

            return false;
        }
    }

    public class DelegateDfsVisitor : DfsVisitor
    {
        private readonly Func<Node, bool> _visitNode;

        public DelegateDfsVisitor(Func<Node, bool> visitNode)
        {
            _visitNode = visitNode;
        }

        public DelegateDfsVisitor(Action<Node> visitNode)
        {
            _visitNode = n =>
            {
                visitNode(n);
                return false;
            };
        }

        protected override bool Visit(Node node)
        {
            return _visitNode(node);
        }
    }
}
