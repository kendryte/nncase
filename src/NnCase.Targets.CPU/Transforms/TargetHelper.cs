using System;
using System.Collections.Generic;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Targets.CPU.Transforms
{
    internal static class TargetHelper
    {
        public static Transpose NHWCToNCHW(this Graph graph, DataType type, Shape shape)
        {
            return graph.AddNode(new Transpose(type, shape, new[] { 0, 3, 1, 2 }));
        }

        public static Transpose NCHWToNHWC(this Graph graph, DataType type, Shape shape)
        {
            return graph.AddNode(new Transpose(type, shape, new[] { 0, 2, 3, 1 }));
        }
    }
}
