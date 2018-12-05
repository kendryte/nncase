using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using TensorFlow;

namespace NnCase.Converter.Model
{
    public class GraphPlanContext
    {
        public Dictionary<InputLayer, TFOutput> Inputs { get; } = new Dictionary<InputLayer, TFOutput>();

        public Dictionary<OutputLayer, TFOutput> Outputs { get; } = new Dictionary<OutputLayer, TFOutput>();

        public Dictionary<OutputConnector, TFOutput> TFOutputs { get; } = new Dictionary<OutputConnector, TFOutput>();

        public Dictionary<Layer, bool> Planning { get; } = new Dictionary<Layer, bool>();

        public TFGraph TFGraph { get; }

        public GraphPlanContext()
        {
            TFGraph = new TFGraph();
        }

        public void Save(Stream stream)
        {
            var buffer = new TFBuffer();
            TFGraph.ToGraphDef(buffer);
            stream.Write(buffer.ToArray());
        }
    }

    public class Graph
    {
        public IReadOnlyList<InputLayer> Inputs { get; }

        public IReadOnlyList<OutputLayer> Outputs { get; }

        public Graph(IReadOnlyList<InputLayer> inputs, IReadOnlyList<OutputLayer> outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        public void Plan(GraphPlanContext context)
        {
            foreach (var layer in Outputs)
                layer.Plan(context);
        }
    }
}
