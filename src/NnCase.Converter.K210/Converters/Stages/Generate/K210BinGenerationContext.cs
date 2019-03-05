using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NnCase.Converter.K210.Converters.Stages.Generate
{
    public class K210BinGenerationContext
    {
        public string Prefix { get; set; }

        public int WeightsBits { get; set; }

        public uint MaxStartAddress { get; set; }

        public uint MainMemoryUsage { get; set; }

        public uint MainMemoryOutputAddress { get; set; }

        public uint MainMemoryOutputSize { get; set; }

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
}
