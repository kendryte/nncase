using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Nncase.SourceGenerator;

public static class RecriverUtil
{
    /// <summary>
    /// check the class attrs.
    /// </summary>
    /// <param name="AttrLists"></param>
    /// <param name="target_attr_name"></param>
    /// <returns></returns>
    public static bool CheckAttributes(SyntaxList<AttributeListSyntax> AttrLists, string target_attr_name)
    {
        foreach (var attributeList in AttrLists)
        {
            foreach (var attr in attributeList.Attributes)
            {
                if (attr is AttributeSyntax { Name: SimpleNameSyntax { Identifier: { ValueText: var cur_attr_name } } }
                && cur_attr_name == target_attr_name)
                {
                    return true;
                }
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
    public static string GetFullName(this NameSyntax name) => name switch
    {
        QualifiedNameSyntax qualifiedName => GetFullName(qualifiedName.Left) + "." + GetFullName(qualifiedName.Right),
        IdentifierNameSyntax identifierName => identifierName.Identifier.ValueText,
        _ => throw new NotSupportedException(name.GetType().Name)
    };

    /// <summary>
    /// check this type is inherit from other type.
    /// </summary>
    /// <param name="typeSymbol"></param>
    /// <param name="baseSymbol"></param>
    /// <returns></returns>
    public static bool IsInheritFrom(this ITypeSymbol? typeSymbol, ITypeSymbol? baseSymbol)
    {
        if (typeSymbol is null || baseSymbol is null)
            return false;
        if (SymbolEqualityComparer.Default.Equals(typeSymbol, baseSymbol))
            return true;
        if (typeSymbol.Name != "object")
            return IsInheritFrom(typeSymbol.BaseType, baseSymbol);
        return false;
    }


    public static DiagnosticDescriptor ClassNotPartialError => new DiagnosticDescriptor(id: "EvalGen001",
                                                                                title: "The Class Must Be partial!",
                                                                                messageFormat: "The Class '{0}' Must Be partial!.",
                                                                                category: "EvaluatorGenerator",
                                                                                DiagnosticSeverity.Error,
                                                                                isEnabledByDefault: true);

    public static DiagnosticDescriptor ClassNotFromInterfaceError => new DiagnosticDescriptor(id: "EvalGen002",
                                                                                title: "The Class Must Be Derived Interface!",
                                                                                messageFormat: "The '{0}' Must Have '{1}'<T>!.",
                                                                                category: "EvaluatorGenerator",
                                                                                DiagnosticSeverity.Error,
                                                                                isEnabledByDefault: true);

    public static DiagnosticDescriptor ClassNoValidMethodError => new DiagnosticDescriptor(id: "EvalGen003",
                                                                                title: "The Class Have No Valid Method!",
                                                                                messageFormat: "The '{0}' Have Not Valid Method!",
                                                                                category: "EvaluatorGenerator",
                                                                                DiagnosticSeverity.Error,
                                                                                isEnabledByDefault: true);

    public static DiagnosticDescriptor ClassMoreMethodError => new DiagnosticDescriptor(id: "EvalGen004",
                                                                            title: "The Class Have More Valid Method!",
                                                                            messageFormat: "The '{0}' Have More Valid Method!",
                                                                            category: "EvaluatorGenerator",
                                                                            DiagnosticSeverity.Error,
                                                                            isEnabledByDefault: true);

    public static DiagnosticDescriptor MethodParamError = new DiagnosticDescriptor(id: "EvalGen005",
                                                                    title: "The Method Parameters Is Not Valid!",
                                                                    messageFormat: "This Class '{0}' Method Parameters Is ('{1}'), Because the `'{2}'` ",
                                                                    category: "EvaluatorGenerator",
                                                                    DiagnosticSeverity.Error,
                                                                    isEnabledByDefault: true);

    public static DiagnosticDescriptor ClassNotFromBaseClassError => new DiagnosticDescriptor(id: "EvalGen006",
                                                                                title: "The Class Must Be Derived From Target Class!",
                                                                                messageFormat: "The '{0}' Must From '{1}'!",
                                                                                category: "EvaluatorGenerator",
                                                                                DiagnosticSeverity.Error,
                                                                                isEnabledByDefault: true);
}
