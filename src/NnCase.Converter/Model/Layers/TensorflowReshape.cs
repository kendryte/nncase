using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.Model.Layers
{
    public class TensorflowReshape : Layer
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        private readonly int[] _newShape;
        public ReadOnlySpan<int> NewShape => _newShape;

        public TensorflowReshape(ReadOnlySpan<int> dimensions, ReadOnlySpan<int> newShape)
        {
            Input = AddInput("input", dimensions);

            _newShape = newShape.ToArray();
            int toDetermIdx = -1;
            int mul = 1;
            for (int i = 0; i < newShape.Length; i++)
            {
                if (newShape[i] == -1)
                    toDetermIdx = i;
                else
                    mul *= newShape[i];
            }

            if (toDetermIdx != -1)
                _newShape[toDetermIdx] = dimensions.GetSize() / mul;

            Output = AddOutput("output", _newShape);
        }

        protected override void OnPlanning(GraphPlanContext context)
        {
            var y = context.TFOutputs[Input.Connection.From];
            var graph = context.TFGraph;

            context.TFOutputs[Output] = context.TFGraph.Reshape(y, context.TFGraph.Const(NewShape.ToNHWC()));
        }
    }
}
