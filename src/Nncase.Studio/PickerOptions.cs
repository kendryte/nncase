// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Avalonia.Platform.Storage;

namespace Nncase.Studio;

public static class PickerOptions
{
    public static FilePickerOpenOptions DataPickerOptions => new FilePickerOpenOptions
    {
        Title = "Open Input File",
        AllowMultiple = true,
        FileTypeFilter = new FilePickerFileType[] { new("npy") { Patterns = new[] { "*.npy" } } },
    };

    public static FilePickerOpenOptions JsonPickerOptions => new FilePickerOpenOptions
    {
        Title = "Open Json File",
        AllowMultiple = true,
        FileTypeFilter = new FilePickerFileType[] { new("json") { Patterns = new[] { "*.json" } } },
    };

    public static FilePickerOpenOptions ImporterPickerOptions => new FilePickerOpenOptions
    {
        Title = "Open Model File",
        AllowMultiple = false,
        FileTypeFilter = new FilePickerFileType[]
        {
            new("model") { Patterns = new[] { "*.tflite", "*.onnx", "*.ncnn" } },
        },
    };

    public static FolderPickerOpenOptions FolderPickerOpenOptions => new FolderPickerOpenOptions()
    {
        Title = "Select Folder",
        AllowMultiple = false,
    };
}
