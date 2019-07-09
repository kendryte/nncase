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

        public static bool TryGetDirectParent<T>(Node node, out T parent)
            where T : Node
        {
            foreach (var input in node.Inputs)
            {
                if (input.Connection?.Owner is T target)
                {
                    parent = target;
                    return true;
                }
            }

            parent = null;
            return false;
        }
    }
}
