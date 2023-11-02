using System;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text.RegularExpressions;

namespace Nncase.Studio;

[AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = true)]
sealed class ValidFloatAttribute : ValidationAttribute
{
    public ValidFloatAttribute()
    {
    }

    public override bool IsValid(object? value)
    {
        if (value == null)
        {
            return false;
        }
        // value
        return float.TryParse((string)value, out var _);
    }
}

[AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = true)]
sealed class ValidFloatArrayAttribute : ValidationAttribute
{
    public ValidFloatArrayAttribute()
    {
    }

    public override bool IsValid(object? value)
    {
        if (value == null)
        {
            return false;
        }
        var s = (((string)value));
        if (s.Contains(","))
        {
            var list = s.Split(",");
            return list.All(s => float.TryParse(s, out var _));
        }
        else
        {
            return float.TryParse(s, out var _);
        }
    }
}

[AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = true)]
sealed class ValidIntArrayAttribute : ValidationAttribute
{
    public ValidIntArrayAttribute()
    {
    }

    public override bool IsValid(object? value)
    {
        if (value == null)
        {
            return false;
        }
        var s = (((string)value));
        if (s.Contains(","))
        {
            var list = s.Split(",");
            return list.All(s => int.TryParse(s, out var _));
        }
        else
        {
            return int.TryParse(s, out var _);
        }
    }
}

[AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = true)]
sealed class ValidLayoutAttribute : ValidationAttribute
{
    // todo: add test
    public ValidLayoutAttribute()
    {
    }

    public override bool IsValid(object? value)
    {
        if (value == null)
        {
            return false;
        }

        // todo: 前处理，5d？？
        var s = (string)value;
        if (s.All(c => "NCHW".Contains(c)))
        {
            return !HasRepeat(s);
        }

        if (TryParseNumberList(s, out var numbers))
        {
            return !HasRepeat(s) && IsSeqNumber(numbers);
        }

        return false;
    }

    private static bool HasRepeat(string s)
    {
        var distinct = s.Distinct().ToHashSet();
        if (distinct.Count != s.Length)
        {
            return true;
        }

        return false;
    }

    private static bool TryParseNumberList(string s, out int[] numbers)
    {
        if (s.Contains(","))
        {
            // todo: all 0-3
            var charArray = s.Split(",").Select(s => s[0]).ToArray();
            return IsNumberList(charArray, out numbers);
        }
        else
        {
            return IsNumberList(s.ToCharArray(), out numbers);
        }
    }

    private static bool IsNumberList(char[] s, out int[] numbers)
    {
        if (s.Any(c => !int.TryParse(c.ToString(), out var _)))
        {
            numbers = Array.Empty<int>();
            return false;
        }

        numbers = s.Select(c => int.Parse(c.ToString())).ToArray();
        return true;
    }

    private static bool IsSeqNumber(int[] seq)
    {
        for (int i = 0; i < seq.Length; i++)
        {
            if (seq[i] != i)
            {
                return false;
            }
        }

        return true;
    }
}
