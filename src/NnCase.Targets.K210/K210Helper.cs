using System;
using System.Collections.Generic;
using System.Text;
using NnCase.IR;

namespace NnCase.Targets.K210
{
    internal class K210Helper
    {
        /// <summary>
        /// 2 MB
        /// </summary>
        public const int KPUMemorySize = 2 * 1024 * 1024;

        public const int KPUMemoryLineSize = 64;

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

        public static int GetBytes(Shape shape)
        {
            return shape[0] * GetBytes(shape[3], shape[2], shape[1]);
        }

        public static int GetBytes(in RuntimeShape shape)
        {
            return shape[0] * GetBytes(shape[3], shape[2], shape[1]);
        }

        public static int GetBytes(int width, int height, int channels)
        {
            (var groups, var rowLength, _) = GetRowLayout(width);
            var oneLineChannels = Math.Min(channels, groups);
            var blocks = (int)Math.Ceiling(channels / (double)oneLineChannels);
            var size = rowLength * height * blocks;
            return size;
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
