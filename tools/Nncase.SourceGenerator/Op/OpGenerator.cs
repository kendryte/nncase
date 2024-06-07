// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.Linq;
using System.Text;
using Humanizer;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;

namespace Nncase.SourceGenerator.Op;

[Generator]
public class OpGenerator : IIncrementalGenerator
{
    private INamedTypeSymbol? _opSymobl;

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        IncrementalValuesProvider<GenerateCandidate> candidates = context.SyntaxProvider
            .CreateSyntaxProvider(
                predicate: static (node, _) => IsSyntaxTargetForGeneration(node),
                transform: (ctx, _) => GetSemanticTargetForGeneration(ctx))
            .Where(static m => m is not null)!;

        context.RegisterSourceOutput(candidates.Collect(), Execute);
    }

    private static bool IsSyntaxTargetForGeneration(SyntaxNode node)
    {
        if (node is ClassDeclarationSyntax { BaseList: { Types.Count: > 0 } } classDeclaration)
        {
            return classDeclaration.Modifiers.Any(tok => tok.IsKind(SyntaxKind.PartialKeyword)) &&
                   !classDeclaration.Modifiers.Any(tok => tok.IsKind(SyntaxKind.AbstractKeyword));
        }

        return false;
    }

    private static void Execute(SourceProductionContext context, ImmutableArray<GenerateCandidate> receiveCandidates)
    {
        var groupedCandidates = receiveCandidates.GroupBy(cand => cand.Op.ContainingNamespace, (IEqualityComparer<ISymbol>)SymbolEqualityComparer.Default).Select(g => (g.Key, g.ToArray()));

        List<NamespaceDeclarationSyntax> namespaces = new();
        foreach (var (old_namespace, candidates) in groupedCandidates)
        {
            List<ClassDeclarationSyntax> classes = new();

            foreach (var cand in candidates)
            {
                var members = new List<MemberDeclarationSyntax>
                {
                    // 1. build ctor
                    ConstructorDeclaration(
                        attributeLists: default,
                        modifiers: TokenList(Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace)),
                        identifier: Identifier(cand.Op.Name),
                        parameterList: ParameterList(SeparatedList(
                            from p in cand.AttrParams
                            select Parameter(
                                attributeLists: default,
                                modifiers: default,
                                type: ParseTypeName(p.Type.ToDisplayString()).WithTrailingTrivia(ElasticSpace),
                                identifier: Identifier(p.Name.Camelize()),
                                @default: default))),
                        initializer: null!,
                        body: GeneratorUtil.MakeBlock(
                            from p in cand.AttrParams
                            select ExpressionStatement(
                                AssignmentExpression(
                                    SyntaxKind.SimpleAssignmentExpression,
                                    MemberAccessExpression(
                                        SyntaxKind.SimpleMemberAccessExpression,
                                        ThisExpression(),
                                        IdentifierName(p.Name)).WithTrailingTrivia(ElasticSpace),
                                    IdentifierName(p.Name.Camelize()).WithLeadingTrivia(ElasticSpace)))))
                    .WithTrailingTrivia(ElasticLineFeed),

                    // 2. build with
                    MethodDeclaration(
                        attributeLists: default,
                        modifiers: TokenList(Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace)),
                        returnType: ParseTypeName(cand.Op.Name).WithTrailingTrivia(ElasticSpace),
                        explicitInterfaceSpecifier: default,
                        identifier: Identifier("With"),
                        typeParameterList: default,
                        parameterList: ParameterList(SeparatedList(
                            from p in cand.AttrParams
                            select Parameter(
                                attributeLists: default,
                                modifiers: default,
                                type: NullableType(ParseTypeName(p.Type.ToDisplayString())).WithTrailingTrivia(ElasticSpace),
                                identifier: Identifier(p.Name.Camelize()),
                                @default: EqualsValueClause(LiteralExpression(SyntaxKind.NullLiteralExpression))))),
                        constraintClauses: default,
                        body: default,
                        expressionBody: ArrowExpressionClause(
                            ImplicitObjectCreationExpression(
                                argumentList: ArgumentList(SeparatedList(
                                    from p in cand.AttrParams
                                    select Argument(
                                        BinaryExpression(
                                            SyntaxKind.CoalesceExpression,
                                            IdentifierName(p.Name.Camelize()).WithTrailingTrivia(ElasticSpace),
                                            MemberAccessExpression(
                                                SyntaxKind.SimpleMemberAccessExpression,
                                                ThisExpression(),
                                                IdentifierName(p.Name)).WithLeadingTrivia(ElasticSpace))))),
                                initializer: default)),
                        semicolonToken: Token(SyntaxKind.SemicolonToken))
                    .WithTrailingTrivia(ElasticLineFeed),

                    // 3. build Equals object
                    MethodDeclaration(
                        attributeLists: default,
                        modifiers: TokenList(
                            Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace),
                            Token(SyntaxKind.OverrideKeyword).WithTrailingTrivia(ElasticSpace)),
                        returnType: PredefinedType(Token(SyntaxKind.BoolKeyword)).WithTrailingTrivia(ElasticSpace),
                        explicitInterfaceSpecifier: default,
                        identifier: Identifier("Equals"),
                        typeParameterList: default,
                        parameterList: ParameterList(SeparatedList(new[] {
                            Parameter(
                                attributeLists: default,
                                modifiers: default,
                                type: NullableType(PredefinedType(Token(SyntaxKind.ObjectKeyword))).WithTrailingTrivia(ElasticSpace),
                                identifier: Identifier("obj"),
                                @default: null),
                        })),
                        constraintClauses: default,
                        body: default,
                        expressionBody: ArrowExpressionClause(
                            InvocationExpression(
                                expression: IdentifierName("Equals"),
                                argumentList: ArgumentList(SeparatedList(new[] {
                                    Argument(
                                        BinaryExpression(
                                            SyntaxKind.AsExpression,
                                            IdentifierName("obj").WithTrailingTrivia(ElasticSpace),
                                            ParseTypeName(cand.Op.Name).WithLeadingTrivia(ElasticSpace))),
                                })))),
                        semicolonToken: Token(SyntaxKind.SemicolonToken))
                    .WithTrailingTrivia(ElasticLineFeed),

                    // 3. build Equals
                    MethodDeclaration(
                        attributeLists: default,
                        modifiers: TokenList(
                            Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace)),
                        returnType: PredefinedType(Token(SyntaxKind.BoolKeyword)).WithTrailingTrivia(ElasticSpace),
                        explicitInterfaceSpecifier: default!,
                        identifier: Identifier("Equals"),
                        typeParameterList: default!,
                        parameterList: ParameterList(SeparatedList(new[] {
                            Parameter(
                                attributeLists: default,
                                modifiers: default,
                                type: NullableType(ParseTypeName(cand.Op.Name)).WithTrailingTrivia(ElasticSpace),
                                identifier: Identifier("other"),
                                @default: null),
                        })),
                        constraintClauses: default,
                        body: Block(
                            IfStatement(
                                InvocationExpression(
                                    IdentifierName("ReferenceEquals"),
                                    ArgumentList(SeparatedList(new[] {
                                        Argument(ThisExpression()),
                                        Argument(IdentifierName("other")),
                                    }))),
                                ReturnStatement(
                                    LiteralExpression(SyntaxKind.TrueLiteralExpression)
                                    .WithLeadingTrivia(ElasticSpace))),
                            ReturnStatement(ChainLogicalAnd(
                                BinaryExpression(
                                    SyntaxKind.LogicalAndExpression,
                                    IsPatternExpression(
                                        IdentifierName("other")
                                            .WithLeadingTrivia(ElasticSpace)
                                            .WithTrailingTrivia(ElasticSpace),
                                        Token(SyntaxKind.IsKeyword).WithTrailingTrivia(ElasticSpace),
                                        UnaryPattern(
                                            Token(SyntaxKind.NotKeyword).WithTrailingTrivia(ElasticSpace),
                                            ConstantPattern(LiteralExpression(SyntaxKind.NullLiteralExpression)))
                                        .WithTrailingTrivia(ElasticSpace))
                                    .WithTrailingTrivia(ElasticSpace),
                                    InvocationExpression(
                                        MemberAccessExpression(
                                            SyntaxKind.SimpleMemberAccessExpression,
                                            BaseExpression(),
                                            IdentifierName("Equals")),
                                        ArgumentList(SeparatedList(new[] {
                                            Argument(IdentifierName("other")),
                                        })))),
                                cand.AttrParams))),
                        semicolonToken: default)
                    .WithTrailingTrivia(ElasticLineFeed),

                    // 3. build GetHashCodeCore
                    MethodDeclaration(
                        attributeLists: default,
                        modifiers: TokenList(
                            Token(SyntaxKind.ProtectedKeyword).WithTrailingTrivia(ElasticSpace),
                            Token(SyntaxKind.OverrideKeyword).WithTrailingTrivia(ElasticSpace)),
                        returnType: PredefinedType(Token(SyntaxKind.IntKeyword)).WithTrailingTrivia(ElasticSpace),
                        explicitInterfaceSpecifier: default,
                        identifier: Identifier("GetHashCodeCore"),
                        typeParameterList: default,
                        parameterList: ParameterList(),
                        constraintClauses: default,
                        body: default,
                        expressionBody: ArrowExpressionClause(
                            InvocationExpression(
                                MemberAccessExpression(
                                    SyntaxKind.SimpleMemberAccessExpression,
                                    IdentifierName("HashCode"),
                                    IdentifierName("Combine")),
                                ArgumentList(SeparatedList(new[] {
                                    Argument(
                                        InvocationExpression(
                                            MemberAccessExpression(
                                                SyntaxKind.SimpleMemberAccessExpression,
                                                BaseExpression(),
                                                IdentifierName("GetHashCodeCore")))),
                                }.Concat(MakeHashCombineExpression(cand.AttrParams)))))),
                        semicolonToken: Token(SyntaxKind.SemicolonToken))
                    .WithTrailingTrivia(ElasticLineFeed),
                };

                classes.Add(GeneratorUtil.MakeClass(cand.Op.Name)
                               .WithModifiers(TokenList(
                                   Token(SyntaxKind.PublicKeyword).WithTrailingTrivia(ElasticSpace),
                                   Token(SyntaxKind.PartialKeyword).WithTrailingTrivia(ElasticSpace)))
                               .WithBaseList(BaseList(
                                   SeparatedList(new BaseTypeSyntax[] {
                                       SimpleBaseType(ParseTypeName($"IEquatable<{cand.Op.Name}?>")),
                                   })))
                    .AddMembers(members.ToArray()));
            }

            // 5. build namespace
            var @namespcae = GeneratorUtil.MakeNameSpace(old_namespace.ToDisplayString())
                .AddMembers(classes.ToArray());
            namespaces.Add(namespcae);
        }

        var compilationUnit = CompilationUnit()
            .WithMembers(new SyntaxList<MemberDeclarationSyntax>(namespaces))
            .WithLeadingTrivia(Comment(Constants.GeneratedFileHeader), GeneratorUtil.MakeWarningTrivid(SyntaxKind.DisableKeyword))
            .WithTrailingTrivia(GeneratorUtil.MakeWarningTrivid(SyntaxKind.RestoreKeyword));

        context.AddSource("Ops", SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText());
    }

    private static IEnumerable<ArgumentSyntax> MakeHashCombineExpression(IEnumerable<IPropertySymbol> attrParams)
    {
        var arguments = new List<ArgumentSyntax>();

        var remain = attrParams.AsEnumerable();
        while (remain.Any())
        {
            var current = remain.Take(7);
            remain = remain.Skip(7);

            arguments.Add(Argument(InvocationExpression(
                MemberAccessExpression(
                    SyntaxKind.SimpleMemberAccessExpression,
                    IdentifierName("HashCode"),
                    IdentifierName("Combine")),
                ArgumentList(SeparatedList(
                    from p in current
                    select Argument(IdentifierName(p.Name)))))));
        }

        return arguments;
    }

    private static BinaryExpressionSyntax ChainLogicalAnd(BinaryExpressionSyntax left, IEnumerable<IPropertySymbol> properties)
    {
        var prop = properties.FirstOrDefault();
        if (prop is not null)
        {
            var inner = BinaryExpression(
                SyntaxKind.LogicalAndExpression,
                left.WithTrailingTrivia(ElasticSpace),
                InvocationExpression(
                    MemberAccessExpression(
                        SyntaxKind.SimpleMemberAccessExpression,
                        IdentifierName(prop.Name),
                        IdentifierName("Equals")),
                    ArgumentList(SeparatedList(new[] {
                        Argument(
                            MemberAccessExpression(
                            SyntaxKind.SimpleMemberAccessExpression,
                            IdentifierName("other"),
                            IdentifierName(prop.Name))),
                    }))));
            return ChainLogicalAnd(inner, properties.Skip(1));
        }

        return left;
    }

    private GenerateCandidate? GetSemanticTargetForGeneration(GeneratorSyntaxContext context)
    {
        _opSymobl ??= context.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.IR.Op");

        var classDeclaration = (ClassDeclarationSyntax)context.Node;
        var op = context.SemanticModel.GetDeclaredSymbol(classDeclaration);

        if (op!.BaseType.IsInheritFrom(_opSymobl))
        {
            IPropertySymbol[] attrParams =
                (from m in classDeclaration.Members.OfType<PropertyDeclarationSyntax>()
                 where m.Modifiers.Any(SyntaxKind.PublicKeyword)
                 && m.AccessorList is { Accessors: { Count: 1 } accessors }
                 && accessors[0].IsKind(SyntaxKind.GetAccessorDeclaration)
                 && accessors[0].SemicolonToken.IsKind(SyntaxKind.SemicolonToken)
                 select context.SemanticModel.GetDeclaredSymbol(m)!).ToArray();
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
}

internal class UsingComparer : IEqualityComparer<UsingDirectiveSyntax>
{
    public bool Equals(UsingDirectiveSyntax x, UsingDirectiveSyntax y)
    {
        return x.Name?.GetFullName() == y.Name?.GetFullName();
    }

    public int GetHashCode(UsingDirectiveSyntax obj)
    {
        return obj.Name?.GetFullName().GetHashCode() ?? 0;
    }
}

internal class GenerateCandidate
{
    public GenerateCandidate(INamedTypeSymbol syb, IPropertySymbol[] attrParams, ISymbol[] exprParams, UsingDirectiveSyntax[] usings)
    {
        Op = syb;
        AttrParams = attrParams;
        ExprParams = exprParams;
        UsingSyntaxs = usings;
    }

    public INamedTypeSymbol Op { get; }

    public IPropertySymbol[] AttrParams { get; }

    public ISymbol[] ExprParams { get; }

    public UsingDirectiveSyntax[] UsingSyntaxs { get; }
}
