// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Onnx;

namespace Nncase.IR;

// internal sealed class ILDotOption
// {
//     private readonly DotNode? _dotNode;
//     private readonly string? _str;

//     public DotNode DotNode => _dotNode!;
//     public string Str => _str!;

//     public bool IsDotNode => _dotNode is not null;

//     public ILDotOption(DotNode dotNode)
//     {
//         _dotNode = dotNode;
//         _str = null;
//     }

//     public ILDotOption(string str)
//     {
//         _dotNode = null;
//         _str = str;
//     }

// }


internal sealed class ILDotPrintVisitor : ExprVisitor<string, string>
{
    private bool display_callable;

    private readonly ModelProto _model;

    private readonly GraphProto _graph;

    public ILDotPrintVisitor(Expr root, bool display_callable)
    {
        this.display_callable = display_callable;
        _model = new();
        _model.ProducerName = "Nncase IL";
        _model.ProducerVersion = "Nncase V2.0";
        _graph = _model.Graph;

        if (root is (TIR.PrimFunction or PrimFunctionWrapper))
            throw new NotSupportedException();

        if (root is BaseFunction basefunc)
        {
            _graph.Name = basefunc.Name;
        }
    }

    public ILDotPrintVisitor(bool display_callable) : this(None.Default, display_callable) { }


    /// <summary>
    /// Save the DotGraph into file
    /// </summary>
    /// <param name="file_path">file path.</param>
    /// <returns>this dot graph.</returns>
    public void SaveToFile(string file_path)
    {
        // if (!file_path.EndsWith(".dot"))
        // {
        //     file_path += ".dot";
        // }

        // var dirName = Path.GetDirectoryName(file_path);
        // if (dirName is not null && dirName != "")
        // {
        //     Directory.CreateDirectory(dirName);
        // }
        // _dotGraph.Build();
        // _dotGraph.SaveToFile(file_path);
        // return _dotGraph;
    }
}
