using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NnCase.IR;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Primitives;

namespace NnCase.Evaluation.Data
{
    public class ImageDataset : Dataset
    {
        public ImageDataset(string path, Shape inputShape, float min, float std)
            : base(path, FilterFileName, inputShape, min, std)
        {
        }

        private static bool FilterFileName(string fileName)
        {
            var formatsManager = Configuration.Default.ImageFormatsManager;
            var format = formatsManager.FindFormatByFileExtension(Path.GetExtension(fileName));
            if (format == null) return false;
            var decoder = formatsManager.FindDecoder(format);
            return decoder != null;
        }

        protected override void Process(ReadOnlySpan<byte> src, Span<float> dest, Shape shape)
        {
            using (var destImage = Decode(src, shape))
            {
                var channelSize = shape[2] * shape[3];

                if (shape[1] == 3)
                {
                    var pixels = destImage.GetPixelSpan();

                    var rChannel = dest.Slice(0, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        rChannel[i] = pixels[i].R / 255.0f;

                    var gChannel = dest.Slice(channelSize, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        gChannel[i] = pixels[i].G / 255.0f;

                    var bChannel = dest.Slice(channelSize * 2, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        bChannel[i] = pixels[i].B / 255.0f;
                }
                else if (shape[1] == 1)
                {
                    destImage.Mutate(x =>
                        x.Grayscale());

                    var pixels = destImage.GetPixelSpan();

                    var rChannel = dest.Slice(0, channelSize);
                    for (int i = 0; i < channelSize; i++)
                        rChannel[i] = pixels[i].R / 255.0f;
                }
                else
                {
                    throw new NotSupportedException($"Channels number {shape[1]} is not supported by image dataset provider.");
                }
            }
        }

        protected Image<Rgb24> Decode(ReadOnlySpan<byte> source, Shape shape)
        {
            var image = Image.Load<Rgb24>(source);
            Image<Rgb24> destImage;
            image.Mutate(x =>
                x.Resize(new ResizeOptions
                {
                    Size = new Size(shape[3], shape[2]),
                    Sampler = KnownResamplers.Bicubic,
                    Mode = ResizeMode.Stretch
                }));
            destImage = image;

            return destImage;
        }
    }
}
