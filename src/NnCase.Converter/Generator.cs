using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Converter
{
    public static class Generator
    {
        public static IEnumerable<double> Step(double start, double stop, int count)
        {
            var step = (stop - start) / (count - 1);
            for (int i = 0; i < count; i++)
                yield return start + step * i;
        }

        public static IEnumerable<int> IntegerStep(int start, int stop, int count)
        {
            var step = (double)(stop - start) / (count - 1);
            for (int i = 0; i < count; i++)
                yield return (int)Math.Floor(start + step * i);
        }
    }
}
