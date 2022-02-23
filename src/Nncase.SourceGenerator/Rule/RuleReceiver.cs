using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Text;

namespace Nncase.SourceGenerator.Rule;


public class RuleReceiver : ISyntaxContextReceiver
{
    public readonly List<Diagnostic> Diagnostics = new();

    private INamedTypeSymbol? ExprSymobl;

    public void OnVisitSyntaxNode(GeneratorSyntaxContext ctx)
    {
        ExprSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.IR.Expr");

        var compilation = ctx.SemanticModel.Compilation;
        if (ctx.Node is ClassDeclarationSyntax classDeclaration)
        {
            var classSymbol = ctx.SemanticModel.GetDeclaredSymbol(classDeclaration);
            if (classSymbol.GetAttributes().Any(attr => attr.AttributeClass.Name == "RuleGeneratorAttribute"))
            {
                if (!classDeclaration.Modifiers.Any(tok => tok.IsKind(SyntaxKind.PartialKeyword)))
                    Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNotPartialError, Location.None, classSymbol.ToDisplayString()));

                // 2. find the reference method!
                var methods = classSymbol.GetMembers().OfType<IMethodSymbol>().Where(m => m.Name == "GetReplace" && m.ReturnType.IsInheritFrom(ExprSymobl!)).ToArray();
                if (methods.Length == 0)
                    return;

                if (methods.Any(m => m.IsOverride && m.Parameters.Length == 1 && m.Parameters[0].Type.Name == "IMatchResult"))
                    return;


                //.Where(m => m.Name == "GetReplace" && m.ReturnType.CheckReturnTypeRange(target_kind)).ToArray();
                //if (methods.Length == 0)
                //    Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNoValidMethodError, Location.None, classSymbol.ToDisplayString()));


                //if (methods.Length > 1)
                //    Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassMoreMethodError, Location.None, classSymbol.ToDisplayString()));

                //var method = methods[0];
                //if (method.ReturnType.Name == target_kind.GetReturnType()
                //                && method.Parameters.Count() == 2
                //                && method.Parameters[0].Type.Name == target_kind.GetContextType()
                //                && method.Parameters[1].Type.Name == OpSymbol.Name)
                //    return;

                //// 3. add to the Candidates
                //Candidates.Add(new(classSymbol, OpSymbol, method, target_kind));
                //Console.WriteLine($"EvaluatorGenerator Receive {classSymbol} For {target_kind}");
            }
        }
    }
}

