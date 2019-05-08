using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.K210.Emulator
{
    public class ForwardContext
    {
        public byte[] KpuRam { get; set; }

        public byte[] MainRam { get; set; }

        public Span<byte> GetKpuRamAt(int address)
        {
            var offset = address * 64;
            return new Span<byte>(KpuRam, offset, KpuRam.Length - offset);
        }

        public Span<byte> GetMainRamAt(int address)
        {
            return new Span<byte>(MainRam, address, MainRam.Length - address);
        }
    }
}
