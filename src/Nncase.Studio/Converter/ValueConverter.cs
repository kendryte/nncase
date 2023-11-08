// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
    // enum to selected index
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value == null)
        {
            return -1;
        }

        var index = Enum.GetNames(value.GetType()).IndexOf(value.ToString());
        return index;
    }

    // to enum
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        // to enum or to other
        if (value == null || (int)value < 0)
        {
            return Activator.CreateInstance(targetType);
        }

        var values = Enum.GetNames(targetType)[(int)value];
        return Enum.Parse(targetType, values);
    }
}
