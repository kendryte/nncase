using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Cli
{
    public class Inference
    {
        public Task Run(Options options)
        {
            switch (options.InputFormat)
            {
                case "tflite":
                default:
                    throw new ArgumentException($"Unsupported input format: {options.InputFormat}");
            }
        }
    }
}
