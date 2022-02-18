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

}
