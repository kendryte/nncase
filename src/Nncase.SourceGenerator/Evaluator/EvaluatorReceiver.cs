using System;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Nncase.SourceGenerator;

/// <summary>
/// the candidate will be generated for new instance
/// </summary>
internal class GenerateCandidate
{
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
internal class EvaluatorImplReceiver : ISyntaxReceiver
{

    /// <summary>
    /// for eval
    /// </summary>
    public readonly Dictionary<string, List<GenerateCandidate>> EvalCandidates = new();

    /// <summary>
    /// for type infer
    /// </summary>
    public readonly Dictionary<string, List<GenerateCandidate>> TypeInferCandidates = new();

    public void OnVisitSyntaxNode(SyntaxNode syntaxNode)
    {
        if (syntaxNode is ClassDeclarationSyntax { BaseList: var base_list, Modifiers: var modifiers } cls
            && base_list is not null
            && modifiers.Any(tok => tok.IsKind(SyntaxKind.PartialKeyword)))
        {
            GenerateCandidate eval_cand = new(cls, null);

            if (CheckFromInterface(eval_cand, InterfaceKind.IEvaluator) && CheckVisitMethod(eval_cand, EvalCandidates, InterfaceKind.IEvaluator))
            {
            }

            GenerateCandidate typeinfer_cand = new(cls, null);
            if (CheckFromInterface(typeinfer_cand, InterfaceKind.ITypeInferencer) && CheckVisitMethod(typeinfer_cand, TypeInferCandidates, InterfaceKind.ITypeInferencer))
            {
            }
        }
    }

    bool CheckFromInterface(GenerateCandidate candidate, InterfaceKind target_interface)
    {
        ClassDeclarationSyntax cls = candidate.classDecl;
        foreach (var baseType in cls.BaseList.Types)
        {
            // check the class is from IEvaluator<Op>, and get the Op 
            if (baseType is SimpleBaseTypeSyntax
                {
                    Type: GenericNameSyntax
                    {
                        Identifier: { ValueText: var cur_interface },
                        TypeArgumentList: { Arguments: var genericArgs }
                    }
                }
                && cur_interface == target_interface.ToString()
                && genericArgs.Count == 1
                && genericArgs[0] is IdentifierNameSyntax { Identifier: { ValueText: var op_type_name } }
            )
            {
                candidate.OpTypeName = op_type_name;
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// check whether the method needs to be modified
    /// </summary>
    /// <param name="candidate">the class op name.</param>
    /// <param name="Candidates">the dict.</param>
    /// <returns></returns>
    bool CheckVisitMethod(GenerateCandidate candidate, Dictionary<string, List<GenerateCandidate>> Candidates, InterfaceKind target_interface)
    {
        var (return_type_name, context_type_name) = target_interface.GetKindInfo();

        /// remove the default interface impl method
        bool RemoveOverride(MemberDeclarationSyntax member)
        {
            if (member is MethodDeclarationSyntax
                {
                    ReturnType: IdentifierNameSyntax { Identifier: { ValueText: var cur_return_type_name } },
                    Identifier: { ValueText: "Visit" },
                    ParameterList: ParameterListSyntax { Parameters: var param },
                } method
                  && cur_return_type_name == return_type_name
                  && param.Count == 2
                  && param[0] is ParameterSyntax { Type: IdentifierNameSyntax { Identifier: { ValueText: var cur_context_name } } }
                  && cur_context_name == context_type_name
                  && param[1] is ParameterSyntax { Type: IdentifierNameSyntax { Identifier: { ValueText: var cur_op_type } } }
                  && cur_op_type == candidate.OpTypeName
            )
                return false;
            return true;
        }

        foreach (var member in candidate.classDecl.Members.Where(RemoveOverride))
        {
            // match the method like public Const Visit(IEvaluateContext context, Celu celu, xxxx)
            if (member is MethodDeclarationSyntax
                {
                    ReturnType: IdentifierNameSyntax { Identifier: { ValueText: var cur_return_type_name } },
                    Identifier: { ValueText: "Visit" },
                    ParameterList: ParameterListSyntax { Parameters: var param }
                } method
                && param.Count > 0
                && cur_return_type_name == return_type_name)
            {
                candidate.methodDecl = method;
                var namespaces = (from namespace_decl in method.Ancestors().OfType<BaseNamespaceDeclarationSyntax>()
                                  select namespace_decl).ToArray();
                if (namespaces.Length != 1)
                    return false;
                var namespace_name = GetFullName(namespaces[0].Name);
                if (!Candidates.TryGetValue(namespace_name, out var member_list))
                {
                    member_list = new() { };
                    Candidates.Add(namespace_name, member_list);
                }
                member_list.Add(candidate);
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// get full name from the name syntax
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    public static string GetFullName(NameSyntax name) => name switch
    {
        QualifiedNameSyntax qualifiedName => GetFullName(qualifiedName.Left) + "." + GetFullName(qualifiedName.Right),
        IdentifierNameSyntax identifierName => identifierName.Identifier.ValueText,
        _ => throw new NotSupportedException(name.GetType().Name)
    };
}