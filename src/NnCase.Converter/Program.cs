using System;
using System.Collections.Generic;
using System.IO;
using NnCase.Converter.Converters;

namespace NnCase.Converter
{
    class Program
    {
        static void Main(string[] args)
        {
            var file = File.ReadAllBytes(@"D:\Work\Repository\models\mobilev1_facenet_optimized.tflite");
            var model = tflite.Model.GetRootAsModel(new FlatBuffers.ByteBuffer(file));
            var tfc = new TfLiteConverter(model, model.Subgraphs(0).Value);
            tfc.Convert();

            Console.WriteLine("Hello World!");
        }
    }
}
