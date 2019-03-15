using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using NnCase.Converter.Model.Layers;
using NnCase.Converter.Transforms;

namespace NnCase.Converter.K210.Transforms
{
    public class K210Conv2dWithMaxAvgPoolTransform : Transform
    {
        protected override bool OnTryMatch(Layer layer, TransformContext context)
        {
            try
            {
                if (layer is K210Conv2d conv2d && conv2d.Conv2dType == K210Conv2dType.Conv2d && conv2d.PoolType == K210PoolType.None)
                {
                    context.MatchedLayers.Add(layer);
                    context.Inputs.Add(conv2d.Input);

                    foreach (var nextLayer in conv2d.Output.Connections.Select(o => o.To.Owner))
                    {
                        if (nextLayer is Dequantize dequantize)
                        {
                            context.MatchedLayers.Add(nextLayer);

                            foreach (var nextLayer2 in dequantize.Output.Connections.Select(o => o.To.Owner))
                            {
                                if (nextLayer2 is MaxPool2d maxPool)
                                {
                                    if (maxPool.FilterWidth != maxPool.FilterHeight ||
                                        maxPool.StrideWidth != maxPool.StrideHeight ||
                                        !((maxPool.FilterWidth == 2 && maxPool.StrideWidth == 2 && (maxPool.Padding == Padding.Valid || NoReminder(maxPool.Input.Dimensions, 2))) ||
                                        (maxPool.FilterWidth == 2 && maxPool.StrideWidth == 1 && maxPool.Padding == Padding.Same) ||
                                        (maxPool.FilterWidth == 4 && maxPool.StrideWidth == 4 && (maxPool.Padding == Padding.Valid || NoReminder(maxPool.Input.Dimensions, 4)))) ||
                                        maxPool.FusedActivationFunction != ActivationFunctionType.Linear)
                                        continue;
                                    context.Outputs.Add(maxPool.Output);
                                }
                                else if (nextLayer2 is AveragePool2d avgPool)
                                {
                                    if (avgPool.FilterWidth != avgPool.FilterHeight ||
                                        avgPool.StrideWidth != avgPool.StrideHeight ||
                                        !((avgPool.FilterWidth == 2 && avgPool.StrideWidth == 2 && (avgPool.Padding == Padding.Valid || NoReminder(avgPool.Input.Dimensions, 2))) ||
                                        (avgPool.FilterWidth == 2 && avgPool.StrideWidth == 1 && avgPool.Padding == Padding.Same) ||
                                        (avgPool.FilterWidth == 4 && avgPool.StrideWidth == 4 && (avgPool.Padding == Padding.Valid || NoReminder(avgPool.Input.Dimensions, 4)))) ||
                                        avgPool.FusedActivationFunction != ActivationFunctionType.Linear)
                                        continue;
                                    context.Outputs.Add(avgPool.Output);
                                }
                                else
                                {
                                    continue;
                                }

                                context.MatchedLayers.Add(nextLayer2);
                                return true;
                            }
                        }
                    }

                    return false;
                }

                return false;
            }
            catch
            {
                return false;
            }
        }

        public override void Process(TransformContext context)
        {
            var conv2d = (K210Conv2d)context.MatchedLayers[0];
            var input = conv2d.Input.Connection.From;
            OutputConnector output;

            conv2d.Input.ClearConnection();

            K210PoolType poolType;
            if (context.MatchedLayers[2] is MaxPool2d maxPool)
            {
                if (maxPool.FilterWidth == 2 && maxPool.StrideWidth == 2)
                    poolType = K210PoolType.MaxPool2x2;
                else if (maxPool.FilterWidth == 2 && maxPool.StrideWidth == 1)
                    poolType = K210PoolType.MaxPool2x2Stride1;
                else if (maxPool.FilterWidth == 4 && maxPool.StrideWidth == 4)
                    poolType = K210PoolType.MaxPool4x4;
                else
                    throw new NotSupportedException("Unsupported max pool.");
                output = maxPool.Output;
            }
            else
            {
                var avgPool = (AveragePool2d)context.MatchedLayers[2];
                if (avgPool.FilterWidth == 2 && avgPool.StrideWidth == 2)
                    poolType = K210PoolType.AveragePool2x2;
                else if (avgPool.FilterWidth == 2 && avgPool.StrideWidth == 1)
                    poolType = K210PoolType.AveragePool2x2Stride1;
                else if (avgPool.FilterWidth == 4 && avgPool.StrideWidth == 4)
                    poolType = K210PoolType.AveragePool4x4;
                else
                    throw new NotSupportedException("Unsupported average pool.");
                output = avgPool.Output;
            }

            var newConv2d = new K210Conv2d(conv2d.Input.Dimensions, conv2d.Conv2dType, conv2d.Weights, conv2d.Bias, poolType, conv2d.FusedActivationFunction, conv2d.NonTrivialActivation);
            newConv2d.Input.SetConnection(input);
            var newDequantize = new Dequantize(newConv2d.Output.Dimensions);
            newDequantize.Input.SetConnection(newConv2d.Output);
            var oldOuts = output.Connections.Select(o => o.To).ToList();
            foreach (var oldOut in oldOuts)
                oldOut.SetConnection(newDequantize.Output);
        }

        private static bool NoReminder(ReadOnlySpan<int> dimensions, int divider)
        {
            return dimensions[2] % divider == 0 && dimensions[3] % divider == 0;
        }
    }
}
