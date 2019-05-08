using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NnCase.Converter.K210.Converters.Stages.Generate
{
    public class K210OutputAddress
    {
        public uint Address { get; set; }

        public uint Size { get; set; }
    }

    public class K210BinGenerationContext
    {
        public string Prefix { get; set; }

        public int WeightsBits { get; set; }

        public uint MaxStartAddress { get; set; }

        public uint MainMemoryUsage { get; set; }

        public IReadOnlyList<K210OutputAddress> Outputs { get; set; }

        public Stream Stream { get; set; }

        public uint AlignStreamPosition(int alignment)
        {
            var cnt = Stream.Position;
            var rem = cnt % alignment;
            if (rem != 0)
                Stream.Position = cnt + (alignment - rem);
            return (uint)Stream.Position;
        }
    }

    public class K210BinDeserializeContext
    {
        public int WeightsBits { get; set; }

        public byte[] KModel { get; set; }

        public SpanReader GetReaderAt(int offset)
        {
            return new SpanReader(new ReadOnlySpan<byte>(KModel, offset, KModel.Length - offset));
        }
    }
}
