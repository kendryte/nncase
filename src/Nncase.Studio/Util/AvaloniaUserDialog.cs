// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using MsBox.Avalonia;

namespace Nncase.Studio.Util;

public class AvaloniaUserDialog
{
    public void ShowDialog(string message) =>
        MessageBoxManager
            .GetMessageBoxStandard("Notification", message).ShowAsync();
}
