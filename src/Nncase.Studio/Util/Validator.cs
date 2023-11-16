// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;
using System.Linq;

namespace Nncase.Studio;

public static class CustomValidator
{
    public static bool ValidateViewModel<T>(T obj, out ICollection<ValidationResult> results)
    {
        results = new List<ValidationResult>();

        return Validator.TryValidateObject(obj!, new ValidationContext(obj!), results, true);
    }
}

[AttributeUsage(AttributeTargets.All, Inherited = false, AllowMultiple = true)]
internal sealed class ValidFloatAttribute : ValidationAttribute
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
internal sealed class ValidFloatArrayAttribute : ValidationAttribute
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

        if (value.GetType() != typeof(string))
        {
            return false;
        }

        var s = (string)value;

        if (s.Contains(",", StringComparison.Ordinal))
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
internal sealed class ValidIntArrayAttribute : ValidationAttribute
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

        if (value.GetType() != typeof(string))
        {
            return false;
        }

        var s = (string)value;
        if (s.Contains(",", StringComparison.Ordinal))
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
internal sealed class ValidLayoutAttribute : ValidationAttribute
{
    public ValidLayoutAttribute()
    {
    }

    public override bool IsValid(object? value)
    {
        if (value == null)
        {
            return false;
        }

        var s = (string)value;
        if (s.All(c => "NCHW".Contains(c, StringComparison.Ordinal)))
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
        if (s.Contains(",", StringComparison.Ordinal))
        {
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
