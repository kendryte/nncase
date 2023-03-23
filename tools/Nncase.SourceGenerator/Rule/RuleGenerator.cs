// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;

// using static Nncase.SourceGenerator.GeneratorUtil;
namespace Nncase.SourceGenerator.Rule;

internal class RuleCandidate
{
    public RuleCandidate(ClassDeclarationSyntax class_declaration, INamedTypeSymbol class_symobl, IMethodSymbol method_symbol)
    {
        ClassDeclaration = class_declaration;
        ClassSymobl = class_symobl;
        MethodSymbol = method_symbol;
    }

    public ClassDeclarationSyntax ClassDeclaration { get; set; }

    public INamedTypeSymbol ClassSymobl { get; set; }

    public IMethodSymbol MethodSymbol { get; set; }
}

[Generator]
internal sealed class RuleGenerator : IIncrementalGenerator
{
    public INamedTypeSymbol? ExprSymobl { get; set; }

    public INamedTypeSymbol? TensorSymobl { get; set; }

    public INamedTypeSymbol? IMatchResultSymobl { get; set; }

    public INamedTypeSymbol? RunPassContextSymobl { get; set; }

    public INamedTypeSymbol? IRewriteRuleSymbol { get; set; }

    public INamedTypeSymbol? QuantRuleSymbol { get; set; }

    // public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new RuleReceiver());
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        // Do a simple filter for enums
        IncrementalValuesProvider<RuleCandidate> candidates = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (s, _) => IsSyntaxTargetForGeneration(s), // select enums with attributes
                transform: (ctx, _) => GetSemanticTargetForGeneration(ctx)) // sect the enum with the [EnumExtensions] attribute
            .Where(static m => m is not null)!; // filter out attributed enums that we don't care about

        // Generate the source using the compilation and enums
        context.RegisterSourceOutput(candidates.Collect(), (spc, source) => Execute(spc, source));
    }

    private static bool IsSyntaxTargetForGeneration(SyntaxNode node)
    {
        return node is ClassDeclarationSyntax classDeclaration && classDeclaration.AttributeLists.Count > 0;
    }

    private RuleCandidate? GetSemanticTargetForGeneration(GeneratorSyntaxContext ctx)
    {
        IRewriteRuleSymbol ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Passes.IRewriteRule")!;
        QuantRuleSymbol ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Passes.QuantRule")!;
        ExprSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.IR.Expr");
        TensorSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Tensor");
        IMatchResultSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.PatternMatch.IMatchResult");
        RunPassContextSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Passes.RunPassContext");

        var classDeclaration = (ClassDeclarationSyntax)ctx.Node;
        var classSymbol = ctx.SemanticModel.GetDeclaredSymbol(classDeclaration);
        if (classSymbol!.GetAttributes().Any(attr => attr.AttributeClass is { Name: "RuleGeneratorAttribute" }))
        {
            // 0. check inherit from base class;
            if (!classSymbol.AllInterfaces.Contains(IRewriteRuleSymbol))
            {
                return null;
            }

            // Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNotFromBaseClassError, Location.None, classSymbol.ToDisplayString(), "RewriteRule"));

            // 1. check is Partial
            if (!classDeclaration.Modifiers.Any(tok => tok.IsKind(SyntaxKind.PartialKeyword)))
            {
                return null;
            }

            // Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNotPartialError, Location.None, classSymbol.ToDisplayString()));

            // 2. find the candidate method!
            var methods = classSymbol.GetMembers().OfType<IMethodSymbol>().Where(m =>
              m.Name == "GetReplace"
              && m.ReturnType.IsInheritFrom(ExprSymobl!)
              && m.Parameters.All(
                  p => SymbolEqualityComparer.Default.Equals(p.Type, IMatchResultSymobl)
                       || SymbolEqualityComparer.Default.Equals(p.Type, RunPassContextSymobl)
                       || p.Type.IsInheritFrom(ExprSymobl) // Expr/ Const / Tuple ...
                       || p.Type.IsInheritFrom(TensorSymobl) // Tensor<?>
                       || (p.Type is INamedTypeSymbol { IsGenericType: true, Name: "IReadOnlyList" } gentype && gentype.TypeArguments[0].IsInheritFrom(ExprSymobl)) // IReadOnlyList<Expr>
                       || p.Type is { IsUnmanagedType: true, IsValueType: true } // int / float ...
                       || p.Type is IArrayTypeSymbol { Rank: 1, ElementType: { IsUnmanagedType: true, IsValueType: true } })) // int[] / float[] ...
                  .ToArray();
            if (methods.Length == 0)
            {
                return null;
            }

            // 3. if have Override method, skip
            if (methods.Any(m =>
                m.IsOverride
                && m.Parameters.Length == 1
                && SymbolEqualityComparer.Default.Equals(m.Parameters[0].Type, IMatchResultSymobl)))
            {
                return null;
            }

            // 4. if have more than one valid method
            if (methods.Length != 1)
            {
                // Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassMoreMethodError, Location.None, classSymbol.ToDisplayString()));
                return null;
            }

            // 5. add to the Candidates
            var method = methods[0];
            return new(classDeclaration, classSymbol, method);
        }

        return null;
    }

    private void Execute(SourceProductionContext context, ImmutableArray<RuleCandidate> candidates)
    {
        // receiver.Diagnostics.ForEach(d => context.ReportDiagnostic(d));
        var grouped_classes = (from cand in candidates
                               select cand.ClassSymobl.ContainingNamespace)
                               .Distinct(SymbolEqualityComparer.Default)
                               .ToDictionary(s => s, s => new List<ClassDeclarationSyntax>(), SymbolEqualityComparer.Default);

        foreach (var cand in candidates)
        {
            // 1. consturct statements
            var statements = new List<StatementSyntax>();
            foreach (var parameterSymbol in cand.MethodSymbol.Parameters)
            {
                string rightExpr;
                switch (parameterSymbol.Type)
                {
                    case INamedTypeSymbol { IsGenericType: true, IsReferenceType: true } x when x.IsInheritFrom(TensorSymobl) && x.Name == "Tensor":
                        rightExpr = $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value.Cast<{x.TypeArguments[0].ToDisplayString()}>()";
                        break;
                    case IArrayTypeSymbol { ElementType: { IsUnmanagedType: true, IsValueType: true } e } x:
                        rightExpr = $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value.ToArray<{e.ToDisplayString()}>()";
                        break;
                    case { IsUnmanagedType: true, IsValueType: true } x:
                        rightExpr = $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value.ToScalar<{x.ToDisplayString()}>()";
                        break;
                    case { IsReferenceType: true } x when x.IsInheritFrom(ExprSymobl):
                        rightExpr = $"({parameterSymbol.Type.ToDisplayString()})__result[\"{parameterSymbol.Name}\"]";
                        break;
                    case ITypeSymbol x when SymbolEqualityComparer.Default.Equals(x, IMatchResultSymobl):
                        rightExpr = $"__result";
                        break;
                    case ITypeSymbol x when SymbolEqualityComparer.Default.Equals(x, RunPassContextSymobl):
                        rightExpr = $"__context";
                        break;
                    case INamedTypeSymbol { IsGenericType: true, Name: "IReadOnlyList" } x when x.TypeArguments[0].IsInheritFrom(ExprSymobl):
                        rightExpr = $"((System.Collections.Generic.IReadOnlyList<Nncase.IR.Expr>)__result[\"{parameterSymbol.Name}\"])";
                        break;
                    case INamedTypeSymbol { IsGenericType: false, IsReferenceType: true } x when SymbolEqualityComparer.Default.Equals(x, TensorSymobl):
                        rightExpr = $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value";
                        break;
                    default:
                        context.ReportDiagnostic(Diagnostic.Create(RecriverUtil.MethodParamError, Location.None, cand.ClassSymobl.Name, parameterSymbol.Name, $"Type {parameterSymbol.Type.ToDisplayString()} Not Support For Generate!"));
                        return;
                }

                statements.Add(
                    ParseStatement($"var {parameterSymbol.Name} = {rightExpr};"));
            }

            if (cand.ClassSymobl.IsInheritFrom(QuantRuleSymbol))
            {
                statements.Add(ParseStatement($"Option = __context;"));
                statements.Add(ParseStatement($"MatchResult = __result;"));
                statements.Add(ParseStatement($"Init();"));
            }

            statements.Add(
              ParseStatement($"return {cand.MethodSymbol.Name}({string.Join(",", cand.MethodSymbol.Parameters.Select(p => p.Name))});"));

            var modifiers = cand.ClassSymobl.BaseType is INamedTypeSymbol baseType && baseType.SpecialType != SpecialType.System_Object
                ? TokenList(Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace), Token(SyntaxKind.OverrideKeyword).WithTrailingTrivia(ElasticSpace))
                : TokenList(Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace));

            // 2. consturct wrapper method.
            var method = MethodDeclaration(ParseTypeName("Nncase.IR.Expr?"), Identifier("GetReplace").WithLeadingTrivia(ElasticSpace))
                        .WithParameterList(ParseParameterList("(IMatchResult __result, RunPassContext __context)"))
                        .WithModifiers(modifiers)
                        .WithBody(Block(statements.Select(s => s
                                .WithLeadingTrivia(ElasticTab)
                                .WithTrailingTrivia(ElasticLineFeed)))
                            .WithLeadingTrivia(ElasticLineFeed)
                            .WithTrailingTrivia(ElasticLineFeed))
                        .WithLeadingTrivia(ElasticTab)
                        .WithTrailingTrivia(ElasticLineFeed);

            // 3. add classes
            grouped_classes[cand.ClassSymobl.ContainingNamespace].Add(
              cand.ClassDeclaration
              .WithIdentifier(Identifier(cand.ClassSymobl.Name))
              .WithMembers(SingletonList<MemberDeclarationSyntax>(method))
              .WithAttributeLists(new SyntaxList<AttributeListSyntax>() { })
              .WithLeadingTrivia(ElasticTab)
              .WithTrailingTrivia(ElasticLineFeed));
        }

        if (grouped_classes.Count == 0)
        {
            return;
        }

        var namespaces = from kv in grouped_classes
                         select GeneratorUtil.MakeNameSpace(kv.Key.ToDisplayString())
                               .AddMembers(kv.Value.ToArray());
        var compilationUnit = CompilationUnit()
            .WithUsings(new(new[]
            {
              GeneratorUtil.MakeUsing("Nncase"),
              GeneratorUtil.MakeUsing("Nncase.IR"),
              GeneratorUtil.MakeUsing("Nncase.PatternMatch"),
            }))
            .WithMembers(new(namespaces))
            .WithLeadingTrivia(Comment(Constants.GeneratedFileHeader), GeneratorUtil.MakeWarningTrivid(SyntaxKind.DisableKeyword))
            .WithTrailingTrivia(GeneratorUtil.MakeWarningTrivid(SyntaxKind.RestoreKeyword));
        context.AddSource("Generated.Rules", SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText());

        // compilation.AddSyntaxTrees(SyntaxTree(compilationUnit, encoding: Encoding.UTF8));
    }
}
