// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen.CPU;

public class KernelArgument
{
    public CSymbol Symbol { get; set; } = null!;
}

public class KernelTemplateModel
{
    public KernelArgument[] Arguments { get; set; } = null!;
}

public class UnaryKernelTemplateModel : KernelTemplateModel
{
    public UnaryOp UnaryOp { get; set; }
}

public class BinaryKernelTemplateModel : KernelTemplateModel
{
    public BinaryOp BinaryOp { get; set; }
}

public class TypedKernelTemplateModel<T> : KernelTemplateModel
    where T : IR.Op
{
    public TypedKernelTemplateModel(T target)
    {
        Target = target;
    }

    public T Target { get; }

    public IR.Expr[] Args { get; set; } = Array.Empty<IR.Expr>();
}
