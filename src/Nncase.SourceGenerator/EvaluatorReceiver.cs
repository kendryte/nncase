using System;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Nncase.SourceGenerator;


/// <summary>
/// get the all class synatx like
/// <code>
/// public partial class CeluEvaluator : IEvaluator<Celu> { 
/// 
/// }
/// </code>
/// </summary>
public class IEvaluatorImplReceiver : ISyntaxReceiver
{
    public class EvalCandidate
    {
        public ClassDeclarationSyntax classDecl;
        public MethodDeclarationSyntax methodDecl;
        public string OpTypeName;
        public string OpParamIdentifier;

        public EvalCandidate(ClassDeclarationSyntax class_decl, MethodDeclarationSyntax method_decl)
        {
            classDecl = class_decl;
            methodDecl = method_decl;
            OpTypeName = "";
            OpParamIdentifier = "";
        }

        public EvalCandidate(ClassDeclarationSyntax class_decl, MethodDeclarationSyntax method_decl, string op_name, string op_param_name)
        {
            classDecl = class_decl;
            methodDecl = method_decl;
            OpTypeName = op_name;
            OpParamIdentifier = op_param_name;
        }
    }

    public readonly Dictionary<string, List<EvalCandidate>> Candidates = new();

    public void OnVisitSyntaxNode(SyntaxNode syntaxNode)
    {
        if (syntaxNode is not ClassDeclarationSyntax cls)
            return;
        CheckFromIEvaluator(new(cls, null));
    }

    void CheckFromIEvaluator(EvalCandidate candidate)
    {
        ClassDeclarationSyntax cls = candidate.classDecl;
        foreach (var baseType in cls.BaseList.Types)
        {
            // check the class is from IEvaluator<Op>, and get the Op 
            if (baseType is SimpleBaseTypeSyntax
                {
                    Type: GenericNameSyntax
                    {
                        Identifier: { ValueText: "IEvaluator" },
                        TypeArgumentList: { Arguments: var genericArgs }
                    }
                }
                && genericArgs.Count == 1
                && genericArgs[0] is IdentifierNameSyntax { Identifier: { ValueText: var op_name } }
            )
            {
                foreach (var member in cls.Members)
                {
                    candidate.OpTypeName = op_name;
                    CheckVisitMethod(candidate, member);
                }
            }
        }
    }

    /// <summary>
    /// check whether the method needs to be modified
    /// </summary>
    /// <param name="candidate">the class op name.</param>
    /// <param name="member"></param>
    /// <returns></returns>
    void CheckVisitMethod(EvalCandidate candidate, MemberDeclarationSyntax member)
    {
        // match the method like public Const Visit(IEvaluateContext context, Celu celu, xxxx)
        if (member is MethodDeclarationSyntax
            {
                ReturnType: IdentifierNameSyntax { Identifier: { ValueText: "Const" } },
                Identifier: { ValueText: "Visit" },
                ParameterList: ParameterListSyntax { Parameters: var param },
                Modifiers:
                {
                    Count: 1,
                } modifiers
            } method
            && param.Count > 2
            && param[0] is ParameterSyntax { Type: IdentifierNameSyntax { Identifier: { ValueText: "IEvaluateContext" } } }
            && param[1] is ParameterSyntax { Identifier: { ValueText: var op_param_name }, Type: IdentifierNameSyntax { Identifier: { ValueText: var op_name } } }
            && op_name == candidate.OpTypeName
            && modifiers[0] == SyntaxFactory.Token(SyntaxKind.PublicKeyword))
        {
            candidate.methodDecl = method;
            candidate.OpParamIdentifier = op_param_name;
            var @namespace = (
              from namespace_decl in method.SyntaxTree.GetRoot().Ancestors().OfType<NamespaceDeclarationSyntax>()
              select namespace_decl).ToArray()[0];
            var namespace_name = GetFullName(@namespace.Name);
            if (!Candidates.TryGetValue(namespace_name, out var member_list))
            {
                member_list = new() { };
                Candidates.Add(namespace_name, member_list);
            }
            member_list.Add(candidate);
        }
    }

    /// <summary>
    /// get full name from the name syntax
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    string GetFullName(NameSyntax name) => name switch
    {
        QualifiedNameSyntax qualifiedName => GetFullName(qualifiedName.Left) + "." + GetFullName(qualifiedName.Right),
        IdentifierNameSyntax identifierName => identifierName.Identifier.ValueText,
        _ => throw new NotSupportedException(name.GetType().Name)
    };
}