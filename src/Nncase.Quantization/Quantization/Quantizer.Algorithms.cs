// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Quantization;

internal partial class Quantizer
{
    private static ValueRange<float> GetMinMax(Tensor<float> tensor)
    {
        var buffer = tensor.Buffer.Span;
        var min = float.MaxValue;
        var max = float.MinValue;

        foreach (var value in buffer)
        {
            if (float.IsFinite(value))
            {
                min = Math.Min(min, value);
                max = Math.Max(max, value);
            }
        }

        return new ValueRange<float>(min, max);
    }

    private static List<float> Smooth(List<float> p, int boxPts = 512)
    {
        var ret = new List<float>(new float[p.Count]);
        var pExpand = new List<float>(new float[boxPts - 1]);
        var pExpand2 = new List<float>(new float[boxPts - 1]);
        pExpand.AddRange(p);
        pExpand.AddRange(pExpand2);

        for (int i = boxPts / 2; i < ret.Count + (boxPts / 2); i++)
        {
            var sum = 0f;
            for (int j = i; j < i + boxPts; j++)
            {
                sum += pExpand[j];
            }

            ret[i - (boxPts / 2)] = sum / boxPts;
        }

        return ret;
    }

    private static List<float> SmoothDistribution(List<float> p, float eps = 0.0001f)
    {
        var isZeros = new List<int>(new int[p.Count]);
        var isNonZeros = new List<int>(new int[p.Count]);
        var nZeros = 0;
        var nNonZeros = 0;
        for (int i = 0; i < p.Count; i++)
        {
            if (p[i] == 0)
            {
                isZeros[i] = 1;
                nZeros++;
                isNonZeros[i] = 0;
            }
            else
            {
                isZeros[i] = 0;
                isNonZeros[i] = 1;
                nNonZeros++;
            }
        }

        if (nNonZeros == 0)
        {
            // The discrete probability distribution is malformed. All entries are 0.
            return new List<float>();
        }

        float eps1 = eps * (float)nZeros / (float)nNonZeros;
        if (eps1 >= 1.0f)
        {
            return new List<float>();
        }

        var ret = p;
        for (int i = 0; i < p.Count; i++)
        {
            ret[i] += (eps * isZeros[i]) - (eps1 * isNonZeros[i]);
        }

        return ret;
    }

    private static float ComputeKld(List<float> p, List<float> q)
    {
        if (p.Count == 0 || q.Count == 0 || p.Count != q.Count)
        {
            return float.MaxValue;
        }

        var pSum = p.Sum();
        var qSum = q.Sum();

        for (int i = 0; i < p.Count; i++)
        {
            p[i] /= pSum;
        }

        for (int i = 0; i < q.Count; i++)
        {
            q[i] /= qSum;
        }

        var d = 0f;
        for (int i = 0; i < p.Count; i++)
        {
            if (p[i] != 0)
            {
                d += q[i] == 0f ? 1f : p[i] * (float)Math.Log((float)(p[i] / q[i]));
            }
        }

        return d;
    }

    private static void GetKldOptRanges(int lowerThreshold, int upperThreshold, int dstBinSize, ref List<float> srcBin, ref float minKld, ref Tuple<int, int> betterThreshold)
    {
        var srcRange = upperThreshold - lowerThreshold;
        var srcPerBin = srcRange / dstBinSize;

        var rangeDist = new List<float>(new float[upperThreshold - lowerThreshold]);
        for (int i = 0; i < upperThreshold - lowerThreshold; i++)
        {
            rangeDist[i] = srcBin[i + lowerThreshold];
        }

        // ref dist
        List<float> refDist = rangeDist;
        for (int i = 0; i < lowerThreshold; i++)
        {
            refDist[0] += srcBin[i];
        }

        for (int i = upperThreshold; i < srcBin.Count; i++)
        {
            refDist[refDist.Count - 1] += srcBin[i];
        }

        // quant dist
        var qDist = new List<float>(new float[dstBinSize]);
        for (int i = 0; i < dstBinSize; i++)
        {
            var start = i * srcPerBin;
            var end = start + srcPerBin;
            var value1 = 0f;

            for (int j = start; j < end; j++)
            {
                value1 += refDist[j];
            }

            qDist[i] = value1;
        }

        // upsample quant dist
        var upsQDist = new List<float>(new float[srcRange]);
        for (int i = 0; i < dstBinSize; i++)
        {
            var start = i * srcPerBin;
            var end = start + srcPerBin;
            var count1 = 0;
            for (int j = start; j < end; j++)
            {
                if (refDist[j] != 0)
                {
                    count1++;
                }
            }

            if (count1 == 0)
            {
                continue;
            }

            var upsampleValue = qDist[i] / count1;
            for (int j = start; j < end; j++)
            {
                if (refDist[j] != 0)
                {
                    upsQDist[j] += upsampleValue;
                }
            }
        }

        var ups2QDist = new List<float>(new float[srcBin.Count]);

        // left outliers
        var count2 = 0;
        for (int i = 0; i < lowerThreshold + srcPerBin; i++)
        {
            if (srcBin[i] != 0)
            {
                count2++;
            }
        }

        var value2 = 0f;
        for (int i = 0; i < lowerThreshold + srcPerBin; i++)
        {
            value2 += srcBin[i];
        }

        value2 /= count2;
        for (int i = 0; i < lowerThreshold + srcPerBin; i++)
        {
            if (srcBin[i] != 0)
            {
                ups2QDist[i] += value2;
            }
        }

        // median
        for (int i = srcPerBin; i < upsQDist.Count - srcPerBin; i++)
        {
            ups2QDist[lowerThreshold + i] = upsQDist[i];
        }

        // right outliers
        count2 = 0;
        for (int i = upperThreshold - srcPerBin; i < srcBin.Count; i++)
        {
            if (srcBin[i] != 0)
            {
                count2++;
            }
        }

        for (int i = upperThreshold - srcPerBin; i < srcBin.Count; i++)
        {
            value2 += srcBin[i];
        }

        value2 /= count2;
        for (int i = upperThreshold - srcPerBin; i < srcBin.Count; i++)
        {
            if (srcBin[i] != 0)
            {
                ups2QDist[i] += value2;
            }
        }

        srcBin = SmoothDistribution(srcBin);
        ups2QDist = SmoothDistribution(ups2QDist);

        float kld = ComputeKld(srcBin, ups2QDist);
        if (kld < minKld)
        {
            minKld = kld;
            betterThreshold = Tuple.Create(lowerThreshold, upperThreshold);
        }

        // return Tuple.Create(betterThreshold, minKld, srcBin);
    }
}
