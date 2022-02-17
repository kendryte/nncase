using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;

namespace Nncase.SourceGenerator.Pattern;



[Generator]
public class PatternGenerator : ISourceGenerator
{
    public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new PatternReceiver());

    public void Execute(GeneratorExecutionContext context)
    {
        if (context.SyntaxContextReceiver is not PatternGenerator patternGenerator)
            return;
    }
}
