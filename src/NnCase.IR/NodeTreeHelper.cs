using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR
{
    public static class NodeTreeHelper
    {
        public static bool TryGetDirectChild<T>(Node node, out T child)
            where T : Node
        {
            foreach (var output in node.Outputs)
            {
                foreach (var conn in output.Connections)
                {
                    if (conn.Owner is T target)
                    {
                        child = target;
                        return true;
                    }
                }
            }

            child = null;
            return false;
        }
    }
}
