// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase;

public enum LSTMDirection
{
    Forward,

    Reverse,

    Bidirectional,
}

public enum LSTMLayout
{
    Zero,

    One,
}

public static class LSTMHelper
{
    public static LSTMDirection ToLSTMDirection(string str)
    {
        return str switch
        {
            "forward" => LSTMDirection.Forward,
            "reverse" => LSTMDirection.Reverse,
            "bidirectional" => LSTMDirection.Bidirectional,
            _ => throw new ArgumentOutOfRangeException(nameof(str), str, null),
        };
    }

    public static LSTMLayout ToLSTMLayout(long n)
    {
        return n switch
        {
            0 => LSTMLayout.Zero,
            1 => LSTMLayout.One,
            _ => throw new ArgumentOutOfRangeException(nameof(n), $"ErrorLSTMLayoutValue:{nameof(n)} Valid value:0/1"),
        };
    }

    public static int LSTMLayoutToValue(LSTMLayout layout)
    {
        return layout switch
        {
            LSTMLayout.Zero => 0,
            LSTMLayout.One => 1,
            _ => throw new ArgumentOutOfRangeException(nameof(layout), layout, null),
        };
    }

    public static string LSTMDirectionToValue(LSTMDirection direction)
    {
        return direction switch
        {
            LSTMDirection.Forward => "forward",
            LSTMDirection.Reverse => "reverse",
            LSTMDirection.Bidirectional => "bidirectional",
            _ => throw new ArgumentOutOfRangeException(nameof(direction), direction, null),
        };
    }
}
