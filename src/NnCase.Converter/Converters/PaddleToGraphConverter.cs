using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using paddle = Paddle.Framework.Proto;

namespace NnCase.Converter.Converters
{
    public class PaddleToGraphConverter
    {
        private readonly string _modelPath;
        private readonly paddle.ProgramDesc _programDesc;
        private readonly Dictionary<InputConnector, int> _inputs;
        private readonly Dictionary<int, OutputConnector> _outputs;

        private paddle.BlockDesc _subgraph;
        public Graph Graph { get; private set; }

        public PaddleToGraphConverter(string modelPath)
        {
            _modelPath = modelPath;
            _programDesc = paddle.ProgramDesc.Parser.ParseFrom(File.ReadAllBytes(Path.Combine(modelPath, "__model__")));

            _inputs = new Dictionary<InputConnector, int>();
            _outputs = new Dictionary<int, OutputConnector>();
        }

        public void Convert(int subgraphIndex)
        {
            _subgraph = _programDesc.Blocks[subgraphIndex];
            var layers = _subgraph.Ops.Select(ConvertOperator).ToList();
        }

        private Layer ConvertOperator(paddle.OpDesc op)
        {
            switch (op.Type)
            {
                case "feed":
                    return ConvertFeed(op);
                case "fetch":
                    return ConvertFetch(op);
                case "conv2d":
                    return ConvertConv2d(op);
                default:
                    throw new NotSupportedException();
            }
        }

        private Layer ConvertFeed(paddle.OpDesc op)
        {
            var output = op.Outputs[0].Arguments[0];
            var layer = new InputLayer(GetVarShape(output));
            return layer;
        }

        private Layer ConvertConv2d(paddle.OpDesc op)
        {
            throw new NotImplementedException();
        }

        private Layer ConvertFetch(paddle.OpDesc op)
        {
            var input = op.Inputs[0].Arguments[0];
            var layer = new OutputLayer(GetVarShape(input));
            return layer;
        }

        private paddle.VarDesc GetVar(string name)
        {
            return _subgraph.Vars.First(o => o.Name == name);
        }

        private int[] GetVarShape(string name)
        {
            var v = GetVar(name);
            return v.Type.LodTensor.Tensor.Dims.Select(x => (int)x).ToArray();
        }
    }
}
