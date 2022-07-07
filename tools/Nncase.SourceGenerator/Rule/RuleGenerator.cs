using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;
//using static Nncase.SourceGenerator.GeneratorUtil;

namespace Nncase.SourceGenerator.Rule;

internal class RuleCandidate
{
    public ClassDeclarationSyntax classDeclaration;
    public INamedTypeSymbol classSymobl;
    public IMethodSymbol methodSymbol;
    public RuleCandidate(ClassDeclarationSyntax class_declaration, INamedTypeSymbol class_symobl, IMethodSymbol method_symbol)
    {
        classDeclaration = class_declaration;
        classSymobl = class_symobl;
        methodSymbol = method_symbol;
    }
}

[Generator]
internal sealed class RuleGenerator : IIncrementalGenerator
{

    public INamedTypeSymbol? ExprSymobl;
    public INamedTypeSymbol? TensorSymobl;
    public INamedTypeSymbol? IMatchResultSymobl;
    public INamedTypeSymbol? RunPassOptionsSymobl;
    public INamedTypeSymbol? IRewriteRuleSymbol;
    public INamedTypeSymbol? QuantRuleSymbol;

    //public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new RuleReceiver());

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

    static bool IsSyntaxTargetForGeneration(SyntaxNode node)
    {
        return node is ClassDeclarationSyntax classDeclaration && classDeclaration.AttributeLists.Count > 0;
    }

    RuleCandidate? GetSemanticTargetForGeneration(GeneratorSyntaxContext ctx)
    {
        IRewriteRuleSymbol ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Transform.IRewriteRule")!;
        QuantRuleSymbol ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Transform.QuantRule")!;
        ExprSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.IR.Expr");
        TensorSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Tensor");
        IMatchResultSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.PatternMatch.IMatchResult");
        RunPassOptionsSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Transform.RunPassOptions");

        var classDeclaration = (ClassDeclarationSyntax)ctx.Node;
        var classSymbol = ctx.SemanticModel.GetDeclaredSymbol(classDeclaration);
        if (classSymbol!.GetAttributes().Any(attr => attr.AttributeClass is { Name: "RuleGeneratorAttribute" }))
        {
            // 0. check inherit from base class;
            if (!classSymbol.AllInterfaces.Contains(IRewriteRuleSymbol))
                return null;
            //Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNotFromBaseClassError, Location.None, classSymbol.ToDisplayString(), "RewriteRule"));

            // 1. check is Partial
            if (!classDeclaration.Modifiers.Any(tok => tok.IsKind(SyntaxKind.PartialKeyword)))
                return null;
            //Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNotPartialError, Location.None, classSymbol.ToDisplayString()));

            // 2. find the candidate method!
            var methods = classSymbol.GetMembers().OfType<IMethodSymbol>().Where(m =>
              m.Name == "GetReplace"
              && m.ReturnType.IsInheritFrom(ExprSymobl!)
              && (m.Parameters.All(
                  p => SymbolEqualityComparer.Default.Equals(p.Type, IMatchResultSymobl)
                       || SymbolEqualityComparer.Default.Equals(p.Type, RunPassOptionsSymobl)
                       || p.Type.IsInheritFrom(ExprSymobl) // Expr/ Const / Tuple ...
                       || p.Type.IsInheritFrom(TensorSymobl) // Tensor<?>
                       || (p.Type is INamedTypeSymbol { IsGenericType: true, Name: "IReadOnlyList" } gentype && gentype.TypeArguments[0].IsInheritFrom(ExprSymobl)) // IReadOnlyList<Expr>
                       || p.Type is { IsUnmanagedType: true, IsValueType: true } // int / float ...
                       || p.Type is IArrayTypeSymbol { Rank: 1, ElementType: { IsUnmanagedType: true, IsValueType: true } } // int[] / float[] ...
                  ))).ToArray();
            if (methods.Length == 0)
                return null;

            // 3. if have Override method, skip
            if (methods.Any(m =>
                m.IsOverride
                && m.Parameters.Length == 1
                && SymbolEqualityComparer.Default.Equals(m.Parameters[0].Type, IMatchResultSymobl)))
                return null;

            // 4. if have more than one valid method 
            if (methods.Length != 1)
                //Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassMoreMethodError, Location.None, classSymbol.ToDisplayString()));
                return null;

            // 5. add to the Candidates
            var method = methods[0];
            return new(classDeclaration, classSymbol, method);
        }
        return null;
    }


    void Execute(SourceProductionContext context, ImmutableArray<RuleCandidate> candidates)
    {
        //receiver.Diagnostics.ForEach(d => context.ReportDiagnostic(d));
        var grouped_classes = (from cand in candidates
                               select cand.classSymobl.ContainingNamespace)
                               .Distinct(SymbolEqualityComparer.Default)
                               .ToDictionary(s => s, s => new List<ClassDeclarationSyntax>(), SymbolEqualityComparer.Default);

        foreach (var cand in candidates)
        {
            // 1. consturct statements
            var statements = new List<StatementSyntax>();
            foreach (var parameterSymbol in cand.methodSymbol.Parameters)
            {
                string rightExpr = parameterSymbol.Type switch
                {
                    INamedTypeSymbol { IsGenericType: true, IsReferenceType: true } x when x.IsInheritFrom(TensorSymobl) && x.Name == "Tensor" => $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value.Cast<{x.TypeArguments[0].ToDisplayString()}>()",
                    IArrayTypeSymbol { ElementType: { IsUnmanagedType: true, IsValueType: true } e } x => $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value.ToArray<{e.ToDisplayString()}>()",
                    { IsUnmanagedType: true, IsValueType: true } x => $"((Nncase.IR.TensorConst)__result[\"{parameterSymbol.Name}\"]).Value.ToScalar<{x.ToDisplayString()}>()",
                    { IsReferenceType: true } x when x.IsInheritFrom(ExprSymobl) => $"({parameterSymbol.Type.ToDisplayString()})__result[\"{parameterSymbol.Name}\"]",
                    ITypeSymbol x when SymbolEqualityComparer.Default.Equals(x, IMatchResultSymobl) => $"__result",
                    ITypeSymbol x when SymbolEqualityComparer.Default.Equals(x, RunPassOptionsSymobl) => $"__options",
                    INamedTypeSymbol { IsGenericType: true, Name: "IReadOnlyList" } x when x.TypeArguments[0].IsInheritFrom(ExprSymobl) => $"((System.Collections.Generic.IReadOnlyList<Nncase.IR.Expr>)__result[\"{parameterSymbol.Name}\"])",
                    _ => throw new NotSupportedException($"Convert {parameterSymbol.Name} {parameterSymbol.Type.ToDisplayString()} For IRewriteRule Impl!")
                };

                statements.Add(
                    ParseStatement($"var {parameterSymbol.Name} = {rightExpr};")
                );
            }
            if (cand.classSymobl.IsInheritFrom(QuantRuleSymbol))
            {
                statements.Add(ParseStatement($"Option = __options;"));
                statements.Add(ParseStatement($"Root = (Expr)__result.Root;"));
                statements.Add(ParseStatement($"Init();"));
            }
            statements.Add(
              ParseStatement($"return {cand.methodSymbol.Name}({string.Join(",", cand.methodSymbol.Parameters.Select(p => p.Name))});")
            );

            var modifiers = cand.classSymobl.BaseType is { IsGenericType: true, Name: "RewriteRule" } || cand.classSymobl.IsInheritFrom(QuantRuleSymbol)
                ? TokenList(Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace), Token(SyntaxKind.OverrideKeyword).WithTrailingTrivia(ElasticSpace))
                : TokenList(Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace));

            // 2. consturct wrapper method.
            var method = MethodDeclaration(ParseTypeName("Nncase.IR.Expr?"), Identifier("GetReplace").WithLeadingTrivia(ElasticSpace))
                        .WithParameterList(ParseParameterList("(IMatchResult __result, RunPassOptions __options)"))
                        .WithModifiers(modifiers)
                        .WithBody(Block(statements.Select(s => s
                                .WithLeadingTrivia(ElasticTab)
                                .WithTrailingTrivia(ElasticLineFeed)))
                            .WithLeadingTrivia(ElasticLineFeed)
                            .WithTrailingTrivia(ElasticLineFeed))
                        .WithLeadingTrivia(ElasticTab)
                        .WithTrailingTrivia(ElasticLineFeed);

            // 3. add classes 
            grouped_classes[cand.classSymobl.ContainingNamespace].Add(
              cand.classDeclaration
              .WithIdentifier(Identifier(cand.classSymobl.Name))
              .WithMembers(SingletonList<MemberDeclarationSyntax>(method))
              .WithAttributeLists(new SyntaxList<AttributeListSyntax>() { })
              .WithLeadingTrivia(ElasticTab)
              .WithTrailingTrivia(ElasticLineFeed)
              );
        }

        if (grouped_classes.Count == 0)
            return;

        var namespaces = (from kv in grouped_classes
                          select GeneratorUtil.MakeNameSpace(kv.Key.ToDisplayString())
                                .AddMembers(kv.Value.ToArray()));

        var compilationUnit = CompilationUnit().
                AddMembers(namespaces.ToArray()).
                AddUsings(
                  GeneratorUtil.MakeUsing("Nncase"),
                  GeneratorUtil.MakeUsing("Nncase.IR"),
                  GeneratorUtil.MakeUsing("Nncase.PatternMatch")
                );
        context.AddSource("Generated.Rules", SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText());
        //compilation.AddSyntaxTrees(SyntaxTree(compilationUnit, encoding: Encoding.UTF8));
    }

}
