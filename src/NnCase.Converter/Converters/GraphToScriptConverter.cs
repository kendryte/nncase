using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model;
using RazorLight;

namespace NnCase.Converter.Converters
{
    public class GraphToScriptConverter
    {
        private readonly Graph _graph;
        private readonly RazorLightEngine _templateEngine;

        public GraphToScriptConverter(Graph graph)
        {
            _graph = graph;
            _templateEngine = new RazorLightEngineBuilder()
                .UseMemoryCachingProvider()
                .UseEmbeddedResourcesProject(typeof(GraphToK210Converter).Assembly, "Templates.Script")
                .Build();
        }

        public async Task ConvertAsync(Graph graph, string outputDir, string prefix)
        {

        }
    }
}
