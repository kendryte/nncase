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
    public readonly List<UsingDirectiveSyntax> UsingDecls = new();
    public string NameSpace;
    public string OpType;
    public readonly List<string> OpAttrParams = new();
    public readonly List<string> OpExprParams = new();


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
    public readonly Dictionary<string, List<GenerateCandidate>> Candidates = new();

    /// <inheritdoc/>
    public void OnVisitSyntaxNode(GeneratorSyntaxContext context)
    {
        // var compilation = context.SemanticModel.Compilation;
        // var recordDecls = context.Node.DescendantNodes().OfType<RecordDeclarationSyntax>();
        // foreach (var recordDecl in recordDecls)
        // {
        //     var recordSyb = context.SemanticModel.GetSymbolInfo(recordDecl);
        // }
        var candidates = new List<GenerateCandidate>();
        if (IsMatch(context.Node, out var candidate))
        {
            candidates.Add(candidate);
        }
    }

    private bool IsMatch(SyntaxNode node, out GenerateCandidate candidate)
    {
        candidate = new();
        if (node is RecordDeclarationSyntax
            {
                AttributeLists: var attrLists,
                BaseList: var baseList,
                Identifier: { ValueText: var opType },
                ParameterList: { Parameters: var attrParamSyntaxs }
            } myrecord
            && RecriverUtil.CheckAttributes(attrLists, "PatternGenerator")
            && RecriverUtil.CheckBaseList(baseList, "Op"))
        {
            //myrecord.Ancestors().OfType<using>
            //attrParamSyntaxs.Select(p => p is
            //{
            //    Type: SimpleNameSyntax
            //    {
            //        Identifier: { ValueText: { var  } }
            //    }
            //});
            var xxx = (from p in attrParamSyntaxs
                       let type = p.Type
                       select p.Identifier.ValueText);

            return true;
        }
        return false;
    }
}