using System;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Nncase.SourceGenerator.Pattern;


/// <summary>
/// the candidate will be generated for new instance
/// </summary>
internal class GenerateCandidate
{
    public List<UsingDirectiveSyntax> UsingDecls;
    public ClassDeclarationSyntax classDecl;
    public MethodDeclarationSyntax methodDecl;
    public string OpTypeName;

    public GenerateCandidate(ClassDeclarationSyntax class_decl, MethodDeclarationSyntax method_decl)
    {
        classDecl = class_decl;
        methodDecl = method_decl;
        OpTypeName = "";
    }

    public GenerateCandidate(ClassDeclarationSyntax class_decl, MethodDeclarationSyntax method_decl, string op_name, string op_param_name)
    {
        classDecl = class_decl;
        methodDecl = method_decl;
        OpTypeName = op_name;
    }
}

/// <summary>
/// collection the evaluator and typeinfer class to candidates.
/// </summary>
internal class PatternReceiver : ISyntaxContextReceiver
{

    INamedTypeSymbol? _irNamespaceSymbol;

    /// <summary>
    /// for eval
    /// </summary>
    public readonly Dictionary<string, List<GenerateCandidate>> EvalCandidates = new();

    /// <inheritdoc/>
    public void OnVisitSyntaxNode(GeneratorSyntaxContext context)
    {
        _irNamespaceSymbol ??= context.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.IR");

        if (_irNamespaceSymbol is not null)
        {
            Console.WriteLine("got some return");
        }
    }
}