using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.IR.Operators
{
    public class Concat : Node
    {
        public OutputConnector Output { get; }

        public int Axis { get; }

        public Concat(DataType dataType, IEnumerable<Shape> inputShapes, int axis)
        {
            Axis = ShapeUtility.NormalizeAxis(inputShapes.First().Count, axis);

            int inputIdx = 0;
            foreach (var shape in inputShapes)
                AddInput("input" + inputIdx++.ToString(), dataType, shape);

            Output = AddOutput("output", dataType, ShapeUtility.GetConcatedShape(inputShapes, Axis));
        }
    }
}
