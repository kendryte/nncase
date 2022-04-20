using System;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Nncase.SourceGenerator.Evaluator;


public enum InterfaceKind
{
    IEvaluator,
    ITypeInferencer,
}

/// <summary>
/// some method get from the interface kink info.
/// </summary>
public static class InterfaceKindExtensions
{
    public static (string return_type_name, string context_type_name) GetKindInfo(this InterfaceKind target_interface) => (target_interface.GetReturnType(), target_interface.GetContextType());

    public static string GetReturnType(this InterfaceKind target_interface) => target_interface switch
    {
        InterfaceKind.IEvaluator => "Nncase.IValue",
        InterfaceKind.ITypeInferencer => "Nncase.IR.IRType",
        _ => throw new NotImplementedException(),
    };

    public static string GetContextType(this InterfaceKind target_interface) => target_interface switch
    {
        InterfaceKind.IEvaluator => "IEvaluateContext",
        InterfaceKind.ITypeInferencer => "ITypeInferenceContext",
        _ => throw new NotImplementedException(),
    };

    public static string GetAttrName(this InterfaceKind target_interface) => target_interface switch
    {
        InterfaceKind.IEvaluator => "EvaluatorGeneratorAttribute",
        InterfaceKind.ITypeInferencer => "TypeInferGeneratorAttribute",
        _ => throw new NotImplementedException(),
    };

    /// <summary>
    /// check the return type , can process the interface type
    /// </summary>
    /// <param name="typeSymbol"></param>
    /// <param name="interfaceKind"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static bool CheckReturnTypeRange(this ITypeSymbol typeSymbol, InterfaceKind interfaceKind) => interfaceKind switch
    {
        InterfaceKind.IEvaluator => typeSymbol switch
        {
            { Name: "Tensor" } or { BaseType: { Name: "Tensor" } } => true,
            { Name: "Const" } or { BaseType: { Name: "Const" } } => true,
            { Name: "IValue" } => true,
            _ => false,
        },
        InterfaceKind.ITypeInferencer => typeSymbol switch
        {
            { Name: "IRType" } => true,
            { BaseType: { Name: "IRType" } } => true,
            _ => false,
        },
        _ => throw new NotImplementedException($"CheckReturnTypeRange : {typeSymbol.Name} {interfaceKind}"),
    };

    public static string BuildReturnWrapper(this ITypeSymbol typeSymbol, InterfaceKind interfaceKind, string visitStatement) => interfaceKind switch
    {
        InterfaceKind.IEvaluator => typeSymbol switch
        {
            { Name: "Tensor" } or { BaseType: { Name: "Tensor" } } => $"Value.FromTensor({visitStatement})",
            { Name: "Const" } or { BaseType: { Name: "Const" } } => $"Value.FromConst({visitStatement})",
            { Name: "IValue" } => visitStatement,
            _ => throw new ArgumentOutOfRangeException($"Can't Return {typeSymbol.ToDisplayString()} For {interfaceKind}!"),
        },
        InterfaceKind.ITypeInferencer => visitStatement,
        _ => throw new NotImplementedException(),
    };
}

/// <summary>
/// the candidate will be generated for new instance
/// </summary>
internal class GenerateCandidate
{
    public INamedTypeSymbol Class;
    public INamedTypeSymbol Op;
    public IMethodSymbol Method;
    public InterfaceKind Target;


    public GenerateCandidate(INamedTypeSymbol classSymbol, INamedTypeSymbol opSymbol, IMethodSymbol method, InterfaceKind target_kind)
    {
        this.Class = classSymbol;
        this.Op = opSymbol;
        this.Method = method;
        this.Target = target_kind;
    }
}

/// <summary>
/// collection the evaluator and typeinfer class to candidates.
/// </summary>
internal class EvaluatorImplReceiver : ISyntaxContextReceiver
{

    /// <summary>
    /// for eval
    /// </summary>
    public readonly List<GenerateCandidate> EvalCandidates = new();

    /// <summary>
    /// for type infer
    /// </summary>
    public readonly List<GenerateCandidate> TypeInferCandidates = new();

    public readonly List<Diagnostic> Diagnostics = new();

    public INamedTypeSymbol? ExprSymobl;
    public INamedTypeSymbol? TensorSymobl;
    public INamedTypeSymbol? ParameterInfoSymobl;
    public INamedTypeSymbol? IRTypeSymobl;
    public INamedTypeSymbol? IEvaluateContextSymobl;
    public INamedTypeSymbol? ITypeInferenceContext;

    public void OnVisitSyntaxNode(GeneratorSyntaxContext ctx)
    {
        ExprSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.IR.Expr");
        TensorSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Tensor");
        IRTypeSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.IR.IRType");
        IEvaluateContextSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Evaluator.IEvaluateContext");
        ITypeInferenceContext ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.Evaluator.ITypeInferenceContext");
        ParameterInfoSymobl ??= ctx.SemanticModel.Compilation.GetTypeByMetadataName("Nncase.IR.ParameterInfo");
        ReceiveTargetInterface(ctx, InterfaceKind.IEvaluator, EvalCandidates);
        ReceiveTargetInterface(ctx, InterfaceKind.ITypeInferencer, TypeInferCandidates);
    }

    void ReceiveTargetInterface(GeneratorSyntaxContext ctx, InterfaceKind target_kind, List<GenerateCandidate> Candidates)
    {
        var compilation = ctx.SemanticModel.Compilation;
        if (ctx.Node is ClassDeclarationSyntax classDeclaration)
        {
            var classSymbol = ctx.SemanticModel.GetDeclaredSymbol(classDeclaration);
            if (classSymbol!.GetAttributes().Any(attr => attr!.AttributeClass!.Name == target_kind.GetAttrName()))
            {
                if (!classDeclaration.Modifiers.Any(tok => tok.IsKind(SyntaxKind.PartialKeyword)))
                {
                    Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNotPartialError, Location.None, classSymbol.ToDisplayString()));
                    return;
                }

                // 1. find op symbol
                var interfaces = classSymbol.Interfaces.Where(i => i.TypeArguments.Count() == 1 && i.Name == target_kind.ToString()).ToArray();
                if (interfaces.Length != 1)
                    Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNotFromInterfaceError, Location.None, classSymbol.ToDisplayString(), target_kind));
                var OpSymbol = interfaces[0].TypeArguments.OfType<INamedTypeSymbol>().First();
                // 2. find the reference method!
                var methods = classSymbol.GetMembers()
                                       .OfType<IMethodSymbol>()
                                       .Where(m => m.Name == "Visit" && m.ReturnType.CheckReturnTypeRange(target_kind)).ToArray();
                if (methods.Length == 0)
                {
                    Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassNoValidMethodError, Location.None, classSymbol.ToDisplayString()));
                    return;
                }


                if (methods.Length > 1)
                {
                    Diagnostics.Add(Diagnostic.Create(RecriverUtil.ClassMoreMethodError, Location.None, classSymbol.ToDisplayString()));
                    return;
                }

                var method = methods[0];
                if (method.ReturnType.Name == target_kind.GetReturnType()
                                && method.Parameters.Count() == 2
                                && method.Parameters[0].Type.Name == target_kind.GetContextType()
                                && method.Parameters[1].Type.Name == OpSymbol.Name)
                    return;

                // 3. add to the Candidates
                Candidates.Add(new(classSymbol, OpSymbol, method, target_kind));
                Console.WriteLine($"EvaluatorGenerator Receive {classSymbol} For {target_kind}");
            }
        }
    }



}