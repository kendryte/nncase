using System;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;
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
            var op = ctx.SemanticModel.GetDeclaredSymbol(recordDeclaration);
            if (op!.BaseType is { Name: "Op" }
              && op!.GetAttributes().Any(attr => attr!.AttributeClass!.Name == "PatternFunctionalGeneratorAttribute")
               )
            {
                var attrParams = (from p in recordDeclaration.ParameterList!.Parameters
                                  select ctx.SemanticModel.GetDeclaredSymbol(p)!).ToArray();
                var exprParams = op.GetMembers()
                    .OfType<IFieldSymbol>()
                    .Where(f => f.Type.Name == "ParameterInfo")
                    .ToArray();
                var unitRoot = ctx.Node.SyntaxTree.GetCompilationUnitRoot();
                var usings = unitRoot.Usings.ToList();
                usings.Add(UsingDirective(ParseName(op.ContainingNamespace.ToDisplayString())));
                Candidates.Add(new(op, attrParams, exprParams, usings.ToArray()));
                Console.WriteLine($"PatternGenerator Receive {op.Name} Op");
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