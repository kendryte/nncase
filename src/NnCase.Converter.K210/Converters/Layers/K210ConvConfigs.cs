using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter.K210.Converters.Layers
{
    public class K210LayerBNConfig
    {
        public int Mul { get; set; }

        public int Shift { get; set; }

        public int Add { get; set; }
    }

    public class K210LayerActConfig
    {
        public long StartX { get; set; }

        public int Mul { get; set; }

        public int Shift { get; set; }

        public int Add { get; set; }
    }

    public class K210ConvLayerConfig
    {
        public ushort[] Weights { get; set; }

        public int ArgX { get; set; }

        public int ShiftX { get; set; }

        public int ArgW { get; set; }

        public int ShiftW { get; set; }

        public long ArgAdd { get; set; }

        public int KernelType { get; set; }

        public int PoolType { get; set; }

        public bool IsDepthwise { get; set; }

        public int InputChannels { get; set; }

        public int OutputChannels { get; set; }

        public K210LayerBNConfig[] BNConfigs { get; set; }

        public K210LayerActConfig[] ActConfigs { get; set; }

        public int InputWidth { get; set; }

        public int InputHeight { get; set; }

        public int OutputWidth { get; set; }

        public int OutputHeight { get; set; }

        public int InputGroups { get; set; }

        public int InputRowLength { get; set; }

        public int OutputGroups { get; set; }

        public int OutputRowLength { get; set; }

        public uint InputAddress { get; set; }

        public uint OutputAddress { get; set; }

        public int OutputChannelsOnTime { get; set; }

        public int LoadTimes { get; set; }

        public int OneLoadKernelsSize { get; set; }

        public int PadValue { get; set; }

        public int InputSize { get; set; }

        public int OutputSize { get; set; }
    }
}
