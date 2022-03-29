using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;
namespace Nncase.SourceGenerator.Rule;

[Generator]
internal class RuleGenerator : ISourceGenerator
{
    public void Execute(GeneratorExecutionContext context)
    {
        if (context.SyntaxContextReceiver is not RuleReceiver receiver)
            return;
        receiver.Diagnostics.ForEach(d => context.ReportDiagnostic(d));
        var grouped_classes = (from cand in receiver.Candidates
                               select cand.classSymobl.ContainingNamespace)
                               .Distinct()
                               .ToDictionary(s => s, s => new List<ClassDeclarationSyntax>());

        foreach (var cand in receiver.Candidates)
        {
            // 1. consturct statements
            var statements = new List<StatementSyntax>();
            foreach (var parameterSymbol in cand.methodSymbol.Parameters)
            {
                string rightExpr = parameterSymbol.Type switch
                {
                    INamedTypeSymbol { IsGenericType: true, IsReferenceType: true } x when x.IsInheritFrom(receiver.TensorSymobl) && x.Name == "Tensor" => $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value.Cast<{x.TypeArguments[0].ToDisplayString()}>()",
                    IArrayTypeSymbol { ElementType: { IsUnmanagedType: true, IsValueType: true } e } x => $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value.ToArray<{e.ToDisplayString()}>()",
                    { IsUnmanagedType: true, IsValueType: true } x => $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value.ToScalar<{x.ToDisplayString()}>()",
                    { IsReferenceType: true } x when x.IsInheritFrom(receiver.ExprSymobl) => $"({ parameterSymbol.Type.ToDisplayString() })__result[\"{parameterSymbol.Name}\"]",
                    ITypeSymbol x when SymbolEqualityComparer.Default.Equals(x, receiver.IMatchResultSymobl) => $"__result",
                    INamedTypeSymbol { IsGenericType: true, Name: "IReadOnlyList" } x when x.TypeArguments[0].IsInheritFrom(receiver.ExprSymobl) => $"((System.Collections.Generic.IReadOnlyList<Nncase.IR.Expr>)__result[\"{parameterSymbol.Name}\"])",
                    _ => throw new NotSupportedException($"Convert {parameterSymbol.Name} {parameterSymbol.Type.ToDisplayString()} For IRewriteRule Impl!")
                };

                statements.Add(
                    ParseStatement($"var {parameterSymbol.Name} = {rightExpr};")
                );
            }
            statements.Add(
              ParseStatement($"return {cand.methodSymbol.Name}({string.Join(",", cand.methodSymbol.Parameters.Select(p => p.Name))});")
            );

            var modifiers = cand.classSymobl.BaseType is { IsGenericType: true, Name: "RewriteRule" }
                ? TokenList(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.OverrideKeyword))
                : TokenList(Token(SyntaxKind.PublicKeyword));

            // 2. consturct wrapper method.
            var method = MethodDeclaration(ParseTypeName("Nncase.IR.Expr?"), Identifier("GetReplace"))
                        .WithParameterList(ParseParameterList("(IMatchResult __result)"))
                        .WithModifiers(modifiers)
                        .WithBody(Block(statements));

            // 3. add classes 
            grouped_classes[cand.classSymobl.ContainingNamespace].Add(
              cand.classDeclaration
              .WithIdentifier(Identifier(cand.classSymobl.Name))
              .WithMembers(SingletonList<MemberDeclarationSyntax>(method))
              .WithAttributeLists(new SyntaxList<AttributeListSyntax>() { })
              );
        }

        if (grouped_classes.Count == 0)
            return;

        var namespaces = (from kv in grouped_classes
                          select NamespaceDeclaration(ParseName(kv.Key.ToDisplayString()))
                                .AddMembers(kv.Value.ToArray()));
        var compilationUnit = CompilationUnit().
                AddMembers(namespaces.ToArray()).
                AddUsings(
                  UsingDirective(ParseName("Nncase")),
                  UsingDirective(ParseName("Nncase.IR")),
                  UsingDirective(ParseName("Nncase.PatternMatch"))
                ).
                NormalizeWhitespace();
        context.AddSource("Generated.Rules", SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText());
    }

    public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new RuleReceiver());

}
