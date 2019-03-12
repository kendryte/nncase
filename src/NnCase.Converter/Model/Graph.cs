using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace NnCase.Converter.Model
{
    public class GraphPlanContext
    {
        public Dictionary<InputLayer, TFOutput> Inputs { get; } = new Dictionary<InputLayer, TFOutput>();

        public Dictionary<OutputLayer, TFOutput> Outputs { get; } = new Dictionary<OutputLayer, TFOutput>();

        public Dictionary<OutputConnector, TFOutput> TFOutputs { get; } = new Dictionary<OutputConnector, TFOutput>();

        public Dictionary<Guid, TFOutput> AdditionalTFOutputs { get; } = new Dictionary<Guid, TFOutput>();

        public Dictionary<Layer, bool> Planning { get; } = new Dictionary<Layer, bool>();

        public TFGraph TFGraph { get; }

        public GraphPlanContext()
        {
            TFGraph = new TFGraph();
        }

        public ValueTask SaveAsync(Stream stream)
        {
            var buffer = new TFBuffer();
            TFGraph.ToGraphDef(buffer);
            return stream.WriteAsync(buffer.ToArray());
        }

        public void Reset()
        {
            Inputs.Clear();
            Outputs.Clear();
            TFOutputs.Clear();
            Planning.Clear();
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
