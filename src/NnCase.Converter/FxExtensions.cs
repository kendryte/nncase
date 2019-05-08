using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NnCase.Converter
{
    public static class FxExtensions
    {
        public static TValue GetValueOrDefault<TKey, TValue>(this IDictionary<TKey, TValue> dict, TKey key)
        {
            if (!dict.TryGetValue(key, out var value))
                value = default(TValue);
            return value;
        }

        public static ValueTask WriteAsync(this Stream stream, byte[] buffer)
        {
            return new ValueTask(stream.WriteAsync(buffer, 0, buffer.Length));
        }

        public static double Clamp(double value, double min, double max)
        {
            return Math.Min(Math.Max(value, min), max);
        }

        public static long Clamp(long value, long min, long max)
        {
            return Math.Min(Math.Max(value, min), max);
        }
    }
}
