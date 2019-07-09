using System;
using System.Collections.Generic;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Kernels;

namespace NnCase.Evaluation.Operators
{
    internal static partial class DefaultEvaulators
    {
        private static void RegisterResizeNearestNeighbor(EvaluatorRegistry registry)
        {
            registry.Add<ResizeNearestNeighbor>((n, e) =>
            {
                var input = e.MemoryAt<byte>(n.Input);
                var output = e.MemoryAt<byte>(n.Output);

                var elementSize = ShapeUtility.GetBytes(n.Input.Type);
                DefaultKernels.ResizeNearestNeighbor(elementSize, input, output, OpUtility.To(n.Input.Shape), n.OutputHeight, n.OutputWidth);
            });
        }

        private static void RegisterResizeBilinear(EvaluatorRegistry registry)
        {
            registry.Add<ResizeBilinear>((n, e) =>
            {
                var input = e.MemoryAt<float>(n.Input);
                var output = e.MemoryAt<float>(n.Output);

                DefaultKernels.ResizeBilinear(input, output, OpUtility.To(n.Input.Shape), n.OutputHeight, n.OutputWidth, n.AlignCorners);
            });
        }
    }
}
