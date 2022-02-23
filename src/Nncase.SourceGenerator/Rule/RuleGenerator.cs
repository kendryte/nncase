using Microsoft.CodeAnalysis;
using System;
using System.Collections.Generic;
using System.Text;

namespace Nncase.SourceGenerator.Rule;

[Generator]
public class RuleGenerator : ISourceGenerator
{
    public void Execute(GeneratorExecutionContext context)
    {

    }

    public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new RuleReceiver());

}
