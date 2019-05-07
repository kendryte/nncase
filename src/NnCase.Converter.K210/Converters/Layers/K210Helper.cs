using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Layers
{
    public static class K210Helper
    {
        public static (int groups, int rowLength, int rowPadding) GetRowLayout(int width)
        {
            int groups, rowLength, rowPadding;

            if (width <= 16)
            {
                groups = 4;
                rowLength = 1;
                rowPadding = 16;
            }
            else if (width <= 32)
            {
                groups = 2;
                rowLength = 1;
                rowPadding = 32;
            }
            else
            {
                groups = 1;
                rowLength = (width + 63) / 64;
                rowPadding = 64;
            }

            return (groups, rowLength, rowPadding);
        }

        public static void KpuUpload(Span<byte> kpuRam, ReadOnlySpan<byte> data, int width, int height, int channels)
        {
            (var groups, var rowLength, var rowPadding) = GetRowLayout(width);
            int srcIdx = 0;
            for (int oc = 0; oc < channels; oc++)
            {
                var channel_origin = oc / groups * rowLength * height * 64 + oc % groups * rowPadding;
                for (int y = 0; y < height; y++)
                {
                    var y_origin = channel_origin + y * rowLength * 64;
                    for (int x = 0; x < width; x++)
                        kpuRam[y_origin + x] = data[srcIdx++];
                }
            }
        }

        public static void KpuDownload(ReadOnlySpan<byte> kpuRam, Span<byte> data, int width, int height, int channels)
        {
            (var groups, var rowLength, var rowPadding) = GetRowLayout(width);
            int srcIdx = 0;
            for (int oc = 0; oc < channels; oc++)
            {
                var channel_origin = oc / groups * rowLength * height * 64 + oc % groups * rowPadding;
                for (int y = 0; y < height; y++)
                {
                    var y_origin = channel_origin + y * rowLength * 64;
                    for (int x = 0; x < width; x++)
                         data[srcIdx++] = kpuRam[y_origin + x];
                }
            }
        }
    }
}
