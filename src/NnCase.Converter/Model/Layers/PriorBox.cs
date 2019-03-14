using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class PriorBox : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Boxes { get; }

        public OutputConnector VariancesOutput { get; }

        public float[] MinSizes { get; }

        public float[] MaxSizes { get; }

        public float[] AspectRatios { get; }

        public float[] Variances { get; }

        public bool Flip { get; }

        public bool Clip { get; }

        public int ImageWidth { get; }

        public int ImageHeight { get; }

        public int StepWidth { get; }

        public int StepHeight { get; }

        public float Offset { get; }

        public PriorBox(ReadOnlySpan<int> dimensions, int imageWidth, int imageHeight, float[] minSizes, float[] maxSizes, float[] aspectRatios, float[] variances, bool flip, bool clip, int stepWidth, int stepHeight, float offset)
        {
            MinSizes = minSizes;
            MaxSizes = maxSizes;
            AspectRatios = aspectRatios;
            Variances = variances;
            Flip = flip;
            Clip = clip;
            ImageWidth = imageWidth;
            ImageHeight = imageHeight;
            StepWidth = stepWidth;
            StepHeight = stepHeight;
            Offset = offset;

            var priorNum = (minSizes.Length * aspectRatios.Length) * (flip ? 2 : 1) + minSizes.Length + maxSizes.Length;
            Input = AddInput("input", dimensions);
            Boxes = AddOutput("boxes", new[] { dimensions[0], dimensions[2] * dimensions[3], priorNum, 4 });
            VariancesOutput = AddOutput("variances", new[] { dimensions[0], dimensions[2] * dimensions[3], priorNum, 4 });
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var graph = context.TFGraph;

            var dummyBoxes = new DenseTensor<float>(Boxes.Dimensions);
            var variances = new DenseTensor<float>(VariancesOutput.Dimensions);
            for (int i = 0; i < variances.Length / 4; i++)
                for (int j = 0; j < 4; j++)
                    variances[i * 4 + j] = Variances[j];

            context.TFOutputs[Boxes] = graph.Const(dummyBoxes.ToTFTensor());
            context.TFOutputs[VariancesOutput] = graph.Const(variances.ToTFTensor());
        }
    }
}
