// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;

namespace Nncase.Runtime.Interop;

public sealed class RTDebugger
{
    public static void Wait(bool b)
    {
        byte value = b switch { true => 1, false => 0, };
        _ = Native.NncaseWaitForDebugger(value);
    }

    public static void Continue()
    {
        _ = Native.NncaseContinueExecution();
    }
}
