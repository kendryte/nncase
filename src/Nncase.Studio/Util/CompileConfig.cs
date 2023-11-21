// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Runtime.InteropServices.JavaScript;
using Nncase.Diagnostics;
using Nncase.Studio.ViewModels;

namespace Nncase.Studio.Util;

public class CompileConfig
{
    public CompileConfig()
    {
        var dumpDir = Path.Join(Directory.GetCurrentDirectory(), "nncase_dump");
        CompileOption.DumpDir = dumpDir;
        CompileOption.InputFile = string.Empty;
        KmodelPath = Path.Join(dumpDir, "test.kmodel");
        ResultDir = dumpDir;
    }

    public CompileOptions CompileOption { get; set; } = new();

    public bool MixQuantize { get; set; }

    public bool UseQuantize { get; set; }

    public PreprocessMode PreprocessMode { get; set; }

    public DumpFlags[] DumpFlags { get; set; } = Array.Empty<DumpFlags>();

    public string Target { get; set; } = "cpu";

    public string KmodelPath { get; set; }

    public string ResultDir { get; set; }

    public string[] InputPathList { get; set; } = Array.Empty<string>();

    public bool EnableShapeBucket
    {
        get { return CompileOption.ShapeBucketOptions.Enable; }
        set { CompileOption.ShapeBucketOptions.Enable = value; }
    }
}
