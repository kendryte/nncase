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

        // Combine the selected enums with the `Compilation`
        IncrementalValueProvider<(Compilation, ImmutableArray<GenerateCandidate>)> compilationAndEnums
            = context.CompilationProvider.Combine(candidates.Collect());

        // Generate the source using the compilation and enums
        context.RegisterSourceOutput(compilationAndEnums,
            static (spc, source) => Execute(source.Item1, source.Item2, spc));
    }

    static bool IsSyntaxTargetForGeneration(SyntaxNode node)
    {
        return (node is RecordDeclarationSyntax { BaseList: BaseListSyntax baseList } record && record.AttributeLists.Count == 1 && baseList.Types.Count == 1);
    }

    static GenerateCandidate? GetSemanticTargetForGeneration(GeneratorSyntaxContext context)
    {
        var recordDeclaration = (RecordDeclarationSyntax)context.Node;
        var op = context.SemanticModel.GetDeclaredSymbol(recordDeclaration);

        if (op!.BaseType is { Name: "Op" }
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

    static void Execute(Compilation compilation, ImmutableArray<GenerateCandidate> receiveCandidates, SourceProductionContext context)
    {
        var groupedCandidates = receiveCandidates.GroupBy(cand => cand.Op.ContainingNamespace, SymbolEqualityComparer.Default).Select(g => (g.Key, g.ToArray()));

        List<NamespaceDeclarationSyntax> namespaces = new();
        foreach (var (old_namespace, candidates) in groupedCandidates)
        {
            List<MemberDeclarationSyntax> members = new List<MemberDeclarationSyntax>();

            var pattern_name_params = new List<ParameterSyntax>()
                {
                    Parameter(Identifier("target_name")).WithType(ParseTypeName("string?")),
                    Parameter(Identifier("call_name")).WithType(ParseTypeName("string?")),
                };
            foreach (var cand in candidates)
            {
                // build the three pattern functional.
                foreach (var name_params in new List<List<ParameterSyntax?>>() { new(){ null,null },
                                                                         new(){ pattern_name_params[0],null },
                                                                    new(){ pattern_name_params[0],pattern_name_params[1] } })
                {
                    { // 1. build normal method
                      // 1.1 method params
                        var method_params = (from p in name_params
                                             where p is not null
                                             select p)
                                     .Concat(from p in cand.AttrParams
                                             select Parameter(Identifier(p.Name)).WithType(ParseTypeName(p.Type.ToDisplayString())))
                                     .Concat(from f in cand.ExprParams
                                             select Parameter(Identifier(f.Name.ToLower())).WithType(ParseTypeName("Pattern")));
                        var statements = new List<StatementSyntax>();
                        {
                            // 1.2 build condition
                            var condition = string.Join("&&", (from p in cand.AttrParams select $"(x.{p.Name} == {p.Name})").DefaultIfEmpty("true"));
                            var inputs = string.Join(", ", from f in cand.ExprParams select f.Name.ToLower());
                            // 1.3 build method return
                            //var x = name_params[0];
                            statements.Add(ParseStatement(@$"return new(
new OpPattern<{cand.Op.ToDisplayString()}>(x => {condition}, {(name_params[0] != null ? "target_name" : "null")}), 
new VArgsPattern (new[]{{ {inputs} }}, null),
{(name_params[1] != null ? "call_name" : "null")});"));
                        }
                        // 1.4. build method body
                        var method = MethodDeclaration(ParseTypeName("CallPattern"), "Is" + cand.Op.Name)
                                     .WithParameterList(ParameterList(SeparatedList(method_params)))
                                     .WithModifiers(TokenList(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.StaticKeyword)))
                                    .WithBody(Block(statements));
                        members.Add(method);
                    }
                    { // 2. build funciton with condition
                      // 2.1 method params
                        var method_params = (from p in name_params
                                             where p is not null
                                             select p)
                                     .Concat(new[] {Parameter(Identifier("condition"))
                                             .WithType(ParseTypeName($"Func<{cand.Op.ToDisplayString()},bool>"))})
                                     .Concat(from f in cand.ExprParams
                                             select Parameter(Identifier(f.Name.ToLower()))
                                             .WithType(ParseTypeName("Pattern")));
                        var statements = new List<StatementSyntax>();
                        {
                            // 1.2 build condition
                            var inputs = string.Join(", ", from f in cand.ExprParams select f.Name.ToLower());
                            // 1.3 build method return
                            statements.Add(ParseStatement(@$"return new(
new OpPattern<{cand.Op.ToDisplayString()}>(condition, {(name_params[0] != null ? "target_name" : "null")}),
new VArgsPattern( new [] {{ {inputs} }}, null ),
{(name_params[1] != null ? "call_name" : "null")});"));
                        }
                        // 1.4. build method body
                        var method = MethodDeclaration(ParseTypeName("CallPattern"), "Is" + cand.Op.Name)
                                     .WithParameterList(ParameterList(SeparatedList(method_params)))
                                     .WithModifiers(TokenList(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.StaticKeyword)))
                                    .WithBody(Block(statements));
                        members.Add(method);
                    }
                }
            }
            // 4. build static class
            List<ClassDeclarationSyntax> classes = new();
            {
                var class_name = old_namespace.MetadataName.Split('.').Last();

                var @class = ClassDeclaration(Identifier(class_name))
                               .WithModifiers(TokenList(
                                   Token(SyntaxKind.PublicKeyword),
                                   Token(SyntaxKind.StaticKeyword),
                                   Token(SyntaxKind.PartialKeyword)))
                               .AddMembers(members.ToArray());
                classes.Add(@class);
            }
            //5. build namespace
            var arr = old_namespace.ToDisplayString().Split('.');
            arr[arr.Length - 1] = "F";
            var @namespcae = NamespaceDeclaration(ParseName(string.Join(".", arr).Replace("IR", "PatternMatch")))
                .AddMembers(classes.ToArray());
            namespaces.Add(namespcae);
        }
        var generatedFiles = CompilationUnit().
                AddMembers(namespaces.ToArray());
        compilation.AddSyntaxTrees(SyntaxTree(generatedFiles, encoding: Encoding.UTF8));
        //context.AddSource("Ops.Pattern", .GetText());
    }
}
