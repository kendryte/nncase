using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter
{
    internal static class Generator
    {
        public static IEnumerable<double> Step(double start, double stop, int count)
        {
            var initial = start;
            double current = start;
            var step = (stop - start) / (count - 1);
            for (int i = 0; i < count; i++)
                yield return start + step * i;
        }
    }
}
