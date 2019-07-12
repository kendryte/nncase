using System;
using System.Collections.Generic;
using System.Text;
using NnCase.IR;

namespace NnCase.Evaluation.Operators
{
    public static class OpUtility
    {
        public static RuntimeShape To(Shape shape)
        {
            ValidateShape(shape);
            var inExt = 4 - shape.Count;
            var rtShape = new RuntimeShape();
            for (int i = 0; i < inExt; i++)
                rtShape[i] = 1;
            for (int i = inExt; i < 4; i++)
                rtShape[i] = shape[i - inExt];
            return rtShape;
        }

        public static RuntimePaddings To(IReadOnlyList<Padding> paddings)
        {
            var inExt = 4 - paddings.Count;
            var rtPaddings = new RuntimePaddings();
            for (int i = 0; i < inExt; i++)
                rtPaddings[i] = Padding.Zero;
            for (int i = inExt; i < 4; i++)
                rtPaddings[i] = paddings[i - inExt];
            return rtPaddings;
        }

        public static (RuntimeShape rtInShape, RuntimeShape rtPerm) ExtendTransposeShape(Shape inShape, Shape perm)
        {
            ValidateShape(inShape);
            ValidateShape(perm);

            RuntimeShape rtInShape;
            RuntimeShape rtPerm;

            var inExt = 4 - inShape.Count;
            var permExt = 4 - perm.Count;
            rtInShape = To(inShape);

            for (int i = 0; i < permExt; i++)
                rtPerm[i] = i;
            for (int i = 0; i < perm.Count; i++)
                rtPerm[i + permExt] = perm[i] + inExt;

            return (rtInShape, rtPerm);
        }

        public static (int innerSize, int outerSize) GetConcatParams(Shape outputShape, int elementSize, int axis)
        {
            int innerSize = elementSize;
            int outerSize = 1;

            for (int i = 0; i < outputShape.Count; i++)
            {
                if (i > axis)
                    innerSize *= outputShape[i];
                else if (i < axis)
                    outerSize *= outputShape[i];
            }

            return (innerSize, outerSize);
        }

        private static void ValidateShape(Shape shape)
        {
            if (shape.Count > 4)
                throw new ArgumentException($"Runtime only support up to 4 rank, but got {shape.Count} rank");
        }
    }
}
