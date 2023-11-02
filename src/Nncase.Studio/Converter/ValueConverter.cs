using System;
using System.Globalization;
using System.Linq;
using Avalonia.Data;
using Avalonia.Data.Converters;
using DynamicData;
using NetFabric.Hyperlinq;

namespace Nncase.Studio;

public class EnumConverter : IValueConverter
{
    // layout
    // NCHW
    // NHWC

    // enum to selected index
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var index = Enum.GetNames(value.GetType()).IndexOf(value.ToString());
        if (index < 0)
        {
            Console.WriteLine("IndexError");
        }

        Console.WriteLine(index);
        Console.WriteLine(value);
        return index;
    }

    // to enum
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        // to enum or to other
        if ((int)value < 0)
        {
            return Activator.CreateInstance(targetType);
        }
        var values = Enum.GetNames(targetType)[(int)value];
        return Enum.Parse(targetType, values);
    }
}

public class ValueConverter : IValueConverter
{
    public static readonly ValueConverter Instance = new();

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value.GetType() == typeof(float))
        {
            return ((float)value).ToString();
        }

        if (value.GetType() == typeof(float[]))
        {
            return string.Join(',', (float[])value);
        }

        return "";
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        Console.WriteLine(value);
        var str = (string)value;
        if (float.TryParse(str, out var res))
        {
            return res;
        }

        if (str == null)
        {
            return Activator.CreateInstance(targetType);
        }

        var list = str.Split(",");
        if (list.All(s => float.TryParse(s, out var _)))
        {
            return list.Select(s => float.Parse(s)).ToArray();
        }

        return new BindingNotification(new InvalidCastException(), BindingErrorType.Error);
    }
    // public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    // {
    //     if (value is string sourceText && parameter is string targetCase
    //                                    && targetType.IsAssignableTo(typeof(string)))
    //     {
    //         switch (targetCase)
    //         {
    //             case "upper":
    //             case "SQL":
    //                 return sourceText.ToUpper();
    //             case "lower":
    //                 return sourceText.ToLower();
    //             case "title": // Every First Letter Uppercase
    //                 var txtinfo = new System.Globalization.CultureInfo("en-US",false).TextInfo;
    //                 return txtinfo.ToTitleCase(sourceText);
    //             default:
    //                 // invalid option, return the exception below
    //                 break;
    //         }
    //     }
    //     // converter used for the wrong type
    //     return new BindingNotification(new InvalidCastException(), BindingErrorType.Error);
    // }
    //
    // public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    // {
    //     return new BindingNotification(new InvalidCastException(), BindingErrorType.Error);
    // }
}
