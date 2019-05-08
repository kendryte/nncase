using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using NnCase.Converter.Converters;
using NnCase.Converter.Data;
using NnCase.Converter.K210.Converters.Layers;
using NnCase.Converter.K210.Converters.Stages.Generate;
using NnCase.Converter.K210.Converters.Stages.Inference;

#if NET471
using System.Collections.Async;
#endif

namespace NnCase.Converter.K210.Emulator
{
    public class K210Emulator
    {
        private const int _outputOffset = 7 * 4;

        private readonly byte[] _kmodel;
        private byte[] _mainMemoryBuffer;
        private byte[] _kpuRam = new byte[2 * 1024 * 1024];
        private int _weightsBits;
        private K210OutputAddress[] _outputAddresses;
        private K210LayerHeader[] _layerHeaders;
        private readonly K210BinDeserializeContext _deserializeContext;

        private int BodyOffset => _outputOffset + _outputAddresses.Length * 4 * 2 + _layerHeaders.Length * 4 * 2;

        public K210Emulator(byte[] kmodel)
        {
            _kmodel = kmodel;
            ReadHeader();
            _deserializeContext = new K210BinDeserializeContext
            {
                WeightsBits = _weightsBits,
                KModel = _kmodel
            };
        }

        public async Task RunAsync(string datasetPath, string outputPath)
        {
            var argument = GetInputArgument();
            var dataset = new ImageDataset(datasetPath, new[] { argument.Config.InputChannels, argument.Config.InputHeight, argument.Config.InputWidth }, 1, PreprocessMethods.None, PostprocessMethods.None);

#if NET471
                await dataset.GetFixedBatchesAsync().ForEachAsync(async batch =>
#else
            await foreach (var batch in dataset.GetFixedBatchesAsync())
#endif
            {
                Run(batch.tensor, argument);
                var outputFile = Path.Combine(outputPath, Path.GetFileNameWithoutExtension(batch.filename[0]) + ".bin");
                using (var bw = new BinaryWriter(File.Open(outputFile, FileMode.Create, FileAccess.Write)))
                {
                    foreach (var outputNode in _outputAddresses)
                    {
                        var buffer = new byte[outputNode.Size];
                        Buffer.BlockCopy(_mainMemoryBuffer, (int)outputNode.Address, buffer, 0, buffer.Length);
                        bw.Write(buffer);
                    }
                }
            }
#if NET471
                );
#endif
        }

        private void Run(Tensor<byte> batch, K210Conv2dLayerArgument inputArgument)
        {
            var context = new ForwardContext { KpuRam = _kpuRam, MainRam = _mainMemoryBuffer };
            var converters = (from t in typeof(K210Emulator).Assembly.ExportedTypes
                              let attrs = t.GetCustomAttributes<LayerConverterAttribute>()
                              where attrs.Any()
                              from attr in attrs
                              where attr.LayerType != K210LayerType.Invalid
                              select new
                              {
                                  Key = attr.LayerType,
                                  Value = new { Type = t, DeserizalizeMethod = t.GetMethod("DeserializeBin"), ForwardMethod = t.GetMethod("Forward") }
                              }).ToDictionary(x => x.Key, x => x.Value);

            KpuInput(inputArgument, context, batch.ToArray());

            var currentBodyOffset = BodyOffset;
            foreach (var layerHeader in _layerHeaders)
            {
                var type = layerHeader.Type;
                if (converters.TryGetValue(type, out var info) && info.DeserizalizeMethod != null)
                {
                    var converter = Activator.CreateInstance(info.Type);
                    var layerArg = info.DeserizalizeMethod.Invoke(converter, new object[] { currentBodyOffset, _deserializeContext });
                    info.ForwardMethod.Invoke(converter, new object[] { layerArg, context });
                }
                else
                {
                    throw new LayerNotSupportedException(type.ToString());
                }

                currentBodyOffset += (int)layerHeader.BodySize;
            }
        }

        private void KpuInput(K210Conv2dLayerArgument inputArgument, ForwardContext context, byte[] data)
        {
            var config = inputArgument.Config;
            K210Helper.KpuUpload(context.GetKpuRamAt((int)config.InputAddress), data, config.InputWidth, config.InputHeight, config.InputChannels);
        }

        private K210Conv2dLayerArgument GetInputArgument()
        {
            if (_layerHeaders[0].Type != K210LayerType.K210Conv)
                throw new InvalidOperationException("The first layer must be k210 conv");

            return new K210Conv2dConverter().DeserializeBin(BodyOffset, _deserializeContext);
        }

        private void ReadHeader()
        {
            var sr = new SpanReader(_kmodel);
            var version = sr.Read<uint>();
            if (version != 3)
                throw new NotSupportedException("Invalid kmodel version");
            var flags = sr.Read<uint>();
            _weightsBits = (flags & 1) == 1 ? 8 : 16;
            var arch = sr.Read<uint>();
            if (arch != 0)
                throw new NotSupportedException("Invalid kmodel arch");
            var layersCount = sr.Read<int>();
            _ = sr.Read<uint>();
            _mainMemoryBuffer = new byte[sr.Read<uint>()];
            var outputCount = sr.Read<int>();

            _outputAddresses = new K210OutputAddress[outputCount];
            for (int i = 0; i < _outputAddresses.Length; i++)
            {
                _outputAddresses[i] = new K210OutputAddress
                {
                    Address = sr.Read<uint>(),
                    Size = sr.Read<uint>()
                };
            }

            _layerHeaders = new K210LayerHeader[layersCount];
            for (int i = 0; i < _layerHeaders.Length; i++)
            {
                _layerHeaders[i] = new K210LayerHeader
                {
                    Type = sr.Read<K210LayerType>(),
                    BodySize = sr.Read<uint>()
                };
            }
        }
    }
}
