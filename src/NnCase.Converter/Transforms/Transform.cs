using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NnCase.Converter.Transforms
{
    public class TransformContext
    {
        public List<Layer> MatchedLayers { get; } = new List<Layer>();

        public List<InputConnector> Inputs { get; } = new List<InputConnector>();

        public List<OutputConnector> Outputs { get; } = new List<OutputConnector>();
    }

    public abstract class Transform
    {
        protected virtual bool SkipSelfContainedCheck => false;

        public bool TryMatch(Layer layer, TransformContext context)
        {
            if (OnTryMatch(layer, context))
            {
                if (!SkipSelfContainedCheck)
                {
                    var inputs = (from l in context.MatchedLayers
                                  from c in l.InputConnectors
                                  where !context.MatchedLayers.Contains(c.Connection?.From.Owner)
                                  select c).Distinct().Except(context.Inputs);
                    if (inputs.Any()) return false;

                    var outputs = (from l in context.MatchedLayers
                                   from c in l.OutputConnectors
                                   from con in c.Connections
                                   where !context.MatchedLayers.Contains(con.To.Owner)
                                   select c).Distinct().Except(context.Outputs);
                    if (outputs.Any()) return false;
                }

                return true;
            }

            return false;
        }

        protected abstract bool OnTryMatch(Layer layer, TransformContext context);

        public abstract void Process(TransformContext context);

        public static void Process(Graph graph, IReadOnlyList<Transform> transforms)
        {
            bool conti = false;

            do
            {
                conti = false;

                foreach (var transform in transforms)
                {
                    bool needRetry = false;
                    do
                    {
                        needRetry = false;
                        var layers = new HashSet<Layer>();
                        foreach (var layer in graph.Outputs)
                            AddAllInputLayer(layer, layers);

                        foreach (var layer in layers)
                        {
                            var context = new TransformContext();
                            if (transform.TryMatch(layer, context))
                            {
                                transform.Process(context);
                                needRetry = true;
                                conti = true;
                                break;
                            }
                        }

                        if (needRetry)
                            RemoveUnusedLayers(graph);
                    } while (needRetry);
                }

            } while (conti);
        }

        private static void RemoveUnusedLayers(Graph graph)
        {
            var usedLayers = new HashSet<Layer>();
            foreach (var layer in graph.Outputs)
                AddUnsedLayer(layer, usedLayers);
            var unusedLayers = new HashSet<Layer>();
            foreach (var layer in graph.Inputs)
                AddAllLayer(layer, unusedLayers);
            unusedLayers.ExceptWith(usedLayers);

            foreach (var unusedLayer in unusedLayers)
                DisconnectLayer(unusedLayer);
        }

        private static void DisconnectLayer(Layer layer)
        {
            foreach (var input in layer.InputConnectors)
                input.ClearConnection();

            foreach (var output in layer.OutputConnectors)
            {
                foreach (var input in output.Connections.ToList())
                    input.To.ClearConnection();
            }
        }

        private static void AddUnsedLayer(Layer layer, HashSet<Layer> layers)
        {
            if (layers.Add(layer))
            {
                foreach (var inputLayer in from c in layer.InputConnectors
                                           where c.Connection != null
                                           select c.Connection.From.Owner)
                    AddUnsedLayer(inputLayer, layers);
            }
        }

        private static void AddAllLayer(Layer layer, HashSet<Layer> layers)
        {
            if (layers.Add(layer))
            {
                foreach (var outputLayer in from c in layer.OutputConnectors
                                            from conn in c.Connections
                                            select conn.To.Owner)
                    AddAllLayer(outputLayer, layers);
            }
        }

        private static void AddAllInputLayer(Layer layer, HashSet<Layer> layers)
        {
            if (layers.Add(layer))
            {
                foreach (var inputLayer in from c in layer.InputConnectors
                                           where c.Connection != null
                                           select c.Connection.From.Owner)
                    AddAllInputLayer(inputLayer, layers);
            }
        }
    }

    public class DummyTransform : Transform
    {
        public override void Process(TransformContext context)
        {
        }

        protected override bool OnTryMatch(Layer layer, TransformContext context) => false;
    }
}
