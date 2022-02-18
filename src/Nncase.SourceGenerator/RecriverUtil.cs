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
    /// check the base list.
    /// </summary>
    /// <param name="baseTypes"></param>
    /// <param name="target_base_type"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static bool CheckBaseList(this BaseListSyntax baseListSyntax, string target_base_type)
    {
        return false;
        // return baseListSyntax.DescendantNodes().OfType<IdentifierNameSyntax>().Any(id => id.Identifier);
        // foreach (var baseType in baseTypes)
        // {
        //     // check the class is from IEvaluator<Op>, and get the Op 
        //     if (baseType is SimpleBaseTypeSyntax
        //         {
        //             Type: IdentifierNameSyntax
        //             {
        //                 Identifier: { ValueText: var cur_base_type }
        //             }
        //         }
        //         && cur_base_type == target_base_type
        //     )
        //     {
        //         return true;
        //     }
        // }
        // throw new ArgumentOutOfRangeException($"This Class Without Target BaseType {target_base_type}");
    }

    // public static bool CheckBaseType()
}
