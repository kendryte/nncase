using System;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Nncase.SourceGenerator.Pattern;



/// <summary>
/// collection the evaluator and typeinfer class to candidates.
/// </summary>
internal class PatternReceiver : ISyntaxContextReceiver
{
    /// <summary>
    /// for eval
    /// </summary>
    public readonly List<GenerateCandidate> Candidates = new();

    /// <inheritdoc/>
    public void OnVisitSyntaxNode(GeneratorSyntaxContext ctx)
    {
        var compilation = ctx.SemanticModel.Compilation;
        if (ctx.Node is RecordDeclarationSyntax recordDeclaration)
        {
            var syb = ctx.SemanticModel.GetDeclaredSymbol(recordDeclaration);
            if (syb.BaseType is { Name: "Op" }
              && syb.GetAttributes().Any(attr => attr.AttributeClass.Name == "PatternGeneratorAttribute")
               )
            {
                var attrParams = (from p in recordDeclaration.ParameterList.Parameters
                                  select ctx.SemanticModel.GetDeclaredSymbol(p)).ToArray();
                var exprParams = syb.GetMembers()
                    .OfType<IFieldSymbol>()
                    .Where(f => f.Type.Name == "ParameterInfo")
                    .ToArray();
                var unitRoot = ctx.Node.SyntaxTree.GetCompilationUnitRoot();
                var usings = unitRoot.Usings.Select(u => ctx.SemanticModel.GetSymbolInfo(u)).ToList();
                Candidates.Add(new(syb, attrParams, exprParams, unitRoot.Usings.ToArray()));
                Console.WriteLine($"PatternGenerator Receive {syb.Name} Op");
            }
        }
    }
}

internal class GenerateCandidate
{
    public INamedTypeSymbol Op;
    public IParameterSymbol[] AttrParams;
    public ISymbol[] ExprParams;
    public UsingDirectiveSyntax[] UsingSyntaxs;

    public GenerateCandidate(INamedTypeSymbol syb, IParameterSymbol[] attrParams, ISymbol[] exprParams, UsingDirectiveSyntax[] usings)
    {
        Op = syb;
        AttrParams = attrParams;
        ExprParams = exprParams;
        UsingSyntaxs = usings;
    }
}