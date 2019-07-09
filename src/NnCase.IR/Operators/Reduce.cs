using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Reduce : Node
    {
        public InputConnector Input { get; }

        public OutputConnector Output { get; }

        public ReduceOperator ReduceOperator { get; }

        public float InitialValue { get; }

        public Shape Axis { get; }

        public bool KeepDims { get; }

        public Reduce(ReduceOperator reduceOperator, float initialValue, Shape inputShape, Shape axis, bool keepDims)
        {
            ReduceOperator = reduceOperator;
            InitialValue = initialValue;
            Axis = ShapeUtility.NormalizeReduceAxis(axis);
            KeepDims = keepDims;

            Input = AddInput("input", DataType.Float32, inputShape);
            Output = AddOutput("output", DataType.Float32, ShapeUtility.GetReducedShape(inputShape, Axis, keepDims));
        }
    }
}
