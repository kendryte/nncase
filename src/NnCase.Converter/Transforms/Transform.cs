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
                                  select c).Except(context.Inputs);
                    if (inputs.Any()) return false;

                    var outputs = (from l in context.MatchedLayers
                                   from c in l.OutputConnectors
                                   from con in c.Connections
                                   where !context.MatchedLayers.Contains(con.To.Owner)
                                   select c).Except(context.Outputs);
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
            var processMap = new Dictionary<Layer, bool>();

            do
            {
                conti = false;

                foreach (var transform in transforms)
                {
                    processMap.Clear();
                    foreach (var layer in graph.Outputs)
                        conti |= Process(layer, transform, processMap);
                }
            } while (conti);
        }

        private static bool Process(Layer layer, Transform transform, Dictionary<Layer, bool> processMap)
        {
            if (processMap.GetValueOrDefault(layer))
                return false;
            processMap[layer] = true;

            bool processed = false;
            var context = new TransformContext();
            if (transform.TryMatch(layer, context))
            {
                transform.Process(context);
                processed = true;
            }

            foreach (var input in layer.InputConnectors)
            {
                if (input.Connection != null)
                    processed |= Process(input.Connection.From.Owner, transform, processMap);
            }

            return processed;
        }
    }
}
