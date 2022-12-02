using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;

namespace Nncase.SourceGenerator.Pattern;

internal class UsingComparer : IEqualityComparer<UsingDirectiveSyntax>
{
    public bool Equals(UsingDirectiveSyntax x, UsingDirectiveSyntax y)
    {
        return x.Name.GetFullName() == y.Name.GetFullName();
    }

    public int GetHashCode(UsingDirectiveSyntax obj)
    {
        return obj.Name.GetFullName().GetHashCode();
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

[Generator]
public class PatternGenerator : IIncrementalGenerator
{
    //public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new PatternReceiver());

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Do a simple filter for enums
        IncrementalValuesProvider<GenerateCandidate> candidates = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) => IsSyntaxTargetForGeneration(node), // select recored with base type named op
                transform: static (ctx, _) => GetSemanticTargetForGeneration(ctx)) // sect the enum with the [EnumExtensions] attribute
            .Where(static m => m is not null)!; // filter out attributed enums that we don't care about

        // Generate the source using the compilation and enums
        context.RegisterSourceOutput(candidates.Collect(), static (spc, source) => Execute(spc, source));
    }

    static bool IsSyntaxTargetForGeneration(SyntaxNode node)
    {
        return (node is RecordDeclarationSyntax { BaseList: BaseListSyntax baseList } record && record.AttributeLists.Count == 1 && baseList.Types.Count == 1);
    }

    static GenerateCandidate? GetSemanticTargetForGeneration(GeneratorSyntaxContext context)
    {
        var recordDeclaration = (RecordDeclarationSyntax)context.Node;
        var op = context.SemanticModel.GetDeclaredSymbol(recordDeclaration);

        if (op!.BaseType is { Name: "Op" or "CustomOp" }
          && op!.GetAttributes().Any(attr => attr!.AttributeClass!.Name == "PatternFunctionalGeneratorAttribute")
           )
        {
            IParameterSymbol[] attrParams = recordDeclaration.ParameterList is null ?
                    new IParameterSymbol[] { } :
                    (from p in recordDeclaration.ParameterList.Parameters
                     select context.SemanticModel.GetDeclaredSymbol(p)!).ToArray();
            var exprParams = op.GetMembers()
                .OfType<IFieldSymbol>()
                .Where(f => f.Type.Name == "ParameterInfo")
                .ToArray();
            var unitRoot = context.Node.SyntaxTree.GetCompilationUnitRoot();
            var usings = unitRoot.Usings.ToList();
            usings.Add(UsingDirective(ParseName(op.ContainingNamespace.ToDisplayString())));
            return new(op, attrParams, exprParams, usings.ToArray());
        }

        return null;
    }

    static void Execute(SourceProductionContext context, ImmutableArray<GenerateCandidate> receiveCandidates)
    {
        var groupedCandidates = receiveCandidates.GroupBy(cand => cand.Op.ContainingNamespace, SymbolEqualityComparer.Default).Select(g => (g.Key, g.ToArray()));

        List<NamespaceDeclarationSyntax> namespaces = new();
        foreach (var (old_namespace, candidates) in groupedCandidates)
        {
            List<MemberDeclarationSyntax> members = new List<MemberDeclarationSyntax>();

            var pattern_name_params = new List<ParameterSyntax>()
                {
                    Parameter(Identifier("target_name")).WithType(ParseTypeName("string?").WithTrailingTrivia(ElasticSpace)),
                    Parameter(Identifier("call_name")).WithType(ParseTypeName("string?").WithTrailingTrivia(ElasticSpace)),
                };
            foreach (var cand in candidates)
            {
                // build the three pattern functional.
                foreach (var name_params in new List<List<ParameterSyntax?>>()
                {
                    new(){ null,null },
                                                                         new(){ pattern_name_params[0],null },
                                                                    new(){ pattern_name_params[0],pattern_name_params[1] }
                })
                {
                    { // 1. build normal method
                      // 1.1 method params
                        var method_params = (from p in name_params
                                             where p is not null
                                             select p)
                                     .Concat(from p in cand.AttrParams
                                             select Parameter(Identifier(p.Name)).
                                                    WithType(
                                                        ParseTypeName(p.Type.ToDisplayString()).
                                                            WithTrailingTrivia(ElasticSpace)))
                                     .Concat(from f in cand.ExprParams
                                             select Parameter(Identifier(f.Name.ToLower()))
                                                    .WithType(ParseTypeName("Pattern ?").WithTrailingTrivia(ElasticSpace))
                                                    .WithDefault(EqualsValueClause(LiteralExpression(SyntaxKind.NullLiteralExpression)))
                                            );
                        var statements = new List<StatementSyntax>();
                        {
                            // 1.2 build condition
                            var condition = string.Join("&&", (from p in cand.AttrParams select $"(x.{p.Name} == {p.Name})").DefaultIfEmpty("true"));
                            var inputs = string.Join(", ", from f in cand.ExprParams select (f.Name.ToLower() + "?? Nncase.PatternMatch.Utility.IsWildcard()"));
                            // 1.3 build method return
                            //var x = name_params[0];
                            statements.Add(ParseStatement(@$"return new(
new OpPattern<{cand.Op.ToDisplayString()}>(x => {condition}, {(name_params[0] != null ? "target_name" : "null")}), 
new VArgsPattern (new[]{{ {inputs} }}, null),
{(name_params[1] != null ? "call_name" : "null")});").
                                           WithLeadingTrivia(ElasticTab).
                                           WithTrailingTrivia(ElasticLineFeed));
                        }

                        // 1.4. build method body
                        var method = GeneratorUtil.MakeMethod(ParseTypeName("CallPattern").WithTrailingTrivia(ElasticSpace), "Is" + cand.Op.Name)
                                     .WithParameterList(ParameterList(SeparatedList(method_params)))
                                     .WithModifiers(TokenList(
                                         Token(SyntaxKind.PublicKeyword)
                                            .WithTrailingTrivia(ElasticSpace),
                                         Token(SyntaxKind.StaticKeyword)
                                            .WithTrailingTrivia(ElasticSpace)))
                                    .WithBody(GeneratorUtil.MakeBlock(statements));
                        members.Add(method);
                    }

                    { // 2. build funciton with condition
                      // 2.1 method params
                        var method_params = (from p in name_params
                                             where p is not null
                                             select p)
                                     .Concat(new[]
                                    {
                                        Parameter(Identifier("condition"))
                                             .WithType(ParseTypeName($"Func<{cand.Op.ToDisplayString()},bool>").WithTrailingTrivia(ElasticSpace))
                                    })
                                     .Concat(from f in cand.ExprParams
                                             select Parameter(Identifier(f.Name.ToLower()))
                                             .WithType(ParseTypeName("Pattern ?").WithTrailingTrivia(ElasticSpace))
                                             .WithDefault(EqualsValueClause(LiteralExpression(SyntaxKind.NullLiteralExpression)))
                                            );
                        var statements = new List<StatementSyntax>();
                        {
                            // 1.2 build condition
                            var inputs = string.Join(", ", from f in cand.ExprParams select (f.Name.ToLower() + "?? Nncase.PatternMatch.Utility.IsWildcard()"));
                            // 1.3 build method return
                            statements.Add(ParseStatement(@$"return new(
new OpPattern<{cand.Op.ToDisplayString()}>(condition, {(name_params[0] != null ? "target_name" : "null")}),
new VArgsPattern( new [] {{ {inputs} }}, null ),
{(name_params[1] != null ? "call_name" : "null")});").
                                           WithLeadingTrivia(ElasticTab).
                                           WithTrailingTrivia(ElasticLineFeed));
                        }

                        // 1.4. build method body
                        var method = GeneratorUtil.MakeMethod(ParseTypeName("CallPattern").WithTrailingTrivia(ElasticSpace), "Is" + cand.Op.Name)
                                     .WithParameterList(ParameterList(SeparatedList(method_params)))
                                     .WithModifiers(
                                        TokenList(
                                            Token(SyntaxKind.PublicKeyword).
                                                WithTrailingTrivia(ElasticSpace),
                                            Token(SyntaxKind.StaticKeyword).
                                                WithTrailingTrivia(ElasticSpace)))
                                    .WithBody(GeneratorUtil.MakeBlock(statements));
                        members.Add(method);
                    }
                }
            }

            // 4. build static class
            List<ClassDeclarationSyntax> classes = new();
            {
                var class_name = old_namespace.MetadataName.Split('.').Last();

                var @class = GeneratorUtil.MakeClass(class_name)
                               .WithModifiers(TokenList(
                                   Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace),
                                   Token(SyntaxKind.StaticKeyword).WithTrailingTrivia(ElasticSpace),
                                   Token(SyntaxKind.PartialKeyword).WithTrailingTrivia(ElasticSpace)))
                    .AddMembers(members.ToArray());
                classes.Add(@class);
            }

            //5. build namespace
            var arr = old_namespace.ToDisplayString().Split('.');
            arr[arr.Length - 1] = "F";
            var @namespcae = GeneratorUtil.MakeNameSpace(string.Join(".", arr).Replace("IR", "PatternMatch"))
                .AddMembers(classes.ToArray());
            namespaces.Add(namespcae);
        }

        var compilationUnit = CompilationUnit().
                WithMembers(new SyntaxList<MemberDeclarationSyntax>(namespaces)).
                WithLeadingTrivia(GeneratorUtil.MakeWarningTrivid(SyntaxKind.DisableKeyword)).
                WithTrailingTrivia(GeneratorUtil.MakeWarningTrivid(SyntaxKind.RestoreKeyword));

        context.AddSource("Ops.Pattern", SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText());
        //compilation.AddSyntaxTrees(SyntaxTree(compilationUnit, encoding: Encoding.UTF8));
    }
}
