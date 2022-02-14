using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;

namespace Nncase.SourceGenerator;

public enum InterfaceKind
{
    IEvaluator,
    ITypeInferencer,
}

public static class InterfaceKindExtension
{
    public static (string return_type_name, string context_type_name) GetKindInfo(this InterfaceKind target_interface) => (target_interface.GetReturnType(), target_interface.GetContextType());

    public static string GetReturnType(this InterfaceKind target_interface) => target_interface switch
    {
        InterfaceKind.IEvaluator => "Const",
        InterfaceKind.ITypeInferencer => "IRType",
        _ => throw new NotImplementedException(),
    };

    public static string GetContextType(this InterfaceKind target_interface) => target_interface switch
    {
        InterfaceKind.IEvaluator => "IEvaluateContext",
        InterfaceKind.ITypeInferencer => "ITypeInferenceContext",
        _ => throw new NotImplementedException(),
    };
}

[Generator]
internal class EvaluatorGenerator : ISourceGenerator
{


    public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new EvaluatorImplReceiver());

    public void Execute(GeneratorExecutionContext context)
    {
        if (context.SyntaxReceiver is not EvaluatorImplReceiver evalReceiver)
            return;

        var eval_compilationunit = BuildFile(context, evalReceiver.EvalCandidates, InterfaceKind.IEvaluator);
        context.AddSource("Ops.Evaluator", SyntaxTree(eval_compilationunit, encoding: Encoding.UTF8).GetText());

        var typeinfer_compilationunit = BuildFile(context, evalReceiver.TypeInferCandidates, InterfaceKind.ITypeInferencer);
        context.AddSource("Ops.TypeInferencer", SyntaxTree(typeinfer_compilationunit, encoding: Encoding.UTF8).GetText());
    }

    /// <summary>
    /// build the value convert expression like:
    /// var alpha = context.GetArgumentValueAsScalar<int>(celu, Celu.Alpha);
    /// </summary>
    /// <param name="cand"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    List<StatementSyntax> BuildStatements(GenerateCandidate cand, InterfaceKind target_interface)
    {
        var (return_type_name, context_type_name) = target_interface.GetKindInfo();
        var statements = new List<StatementSyntax>();
        TextInfo myTI = new CultureInfo("en-US", false).TextInfo;
        foreach (var param in cand.methodDecl.ParameterList.Parameters)
        {
            var paramName = param.Identifier.ValueText;
            string callMethod = param.Type switch
            {
                PredefinedTypeSyntax predefinedType => target_interface switch
                {
                    InterfaceKind.IEvaluator => $"GetArgumentValueAsScalar<{predefinedType.Keyword.ValueText}>",
                    InterfaceKind.ITypeInferencer => throw new NotSupportedException("The Type Infer Can't Convert Predefined Type"),
                    _ => throw new NotSupportedException($"The PredefinedType {predefinedType.Keyword.ValueText}")
                },
                QualifiedNameSyntax qualified => EvaluatorImplReceiver.GetFullName(qualified) switch
                {
                    var x when target_interface == InterfaceKind.IEvaluator && x.EndsWith("torch.Tensor") => "GetTorchArgumentValue",
                    var x when target_interface == InterfaceKind.IEvaluator && x.EndsWith("Tensorflow.Tensor") => "GetTFArgumentValue",
                    var x when target_interface == InterfaceKind.ITypeInferencer => $"CheckArgumentType<{x}>",
                    var x when x == cand.OpTypeName => "",
                    var x => throw new NotSupportedException(x)
                },
                SimpleNameSyntax nameSyntax => nameSyntax switch
                {
                    var x when x.Identifier.ValueText == context_type_name => "",
                    var x when target_interface == InterfaceKind.ITypeInferencer => $"CheckArgumentType<{x}>",
                    _ => throw new NotSupportedException($"When Generate {target_interface.ToString()} Not Support {nameSyntax}")
                },
                _ => throw new NotSupportedException($"The param.Type {param.Type.ToString()}")
            };
            if (callMethod != string.Empty)
                statements.Add(
                  ParseStatement($"var {paramName} = context.{callMethod}({cand.OpTypeName.ToLower()}, {cand.OpTypeName}.{myTI.ToTitleCase(paramName)});")
                );
        }
        statements.Add(
          ParseStatement($"return Visit({string.Join(",", cand.methodDecl.ParameterList.Parameters.Select(p => p.Identifier.ValueText))});")
        );
        return statements;
    }

    /// <summary>
    /// build the whole call method like:
    /// <code>
    /// public Const Visit(IEvaluateContext context, Celu celu)
    /// {
    ///     var input = context.GetTorchArgumentValue(celu, Celu.Input);
    ///     var alpha = context.GetArgumentValueAsScalar<int>(celu, Celu.Alpha);
    ///     return Visit(context, celu, input, alpha);
    /// }
    /// </code>
    /// </summary>
    /// <param name="cand"></param>
    /// <param name="statements"></param>
    /// <returns></returns>
    MethodDeclarationSyntax BuildMethod(GenerateCandidate cand, List<StatementSyntax> statements, InterfaceKind target_interface)
    {
        var (return_type_name, context_type_name) = target_interface.GetKindInfo();
        var method = MethodDeclaration(ParseTypeName(return_type_name), "Visit")
        .AddModifiers(Token(SyntaxKind.PublicKeyword))
        .AddParameterListParameters(
          Parameter(Identifier("context")).WithType(ParseTypeName(context_type_name)),
          Parameter(Identifier(cand.OpTypeName.ToLower())).WithType(ParseTypeName(cand.OpTypeName)))
        .WithBody(Block(statements.ToArray()));
        return method;
    }

    CompilationUnitSyntax BuildFile(GeneratorExecutionContext context, Dictionary<string, List<GenerateCandidate>> Candidates, InterfaceKind target_interface)
    {
        var namespaces = new List<NamespaceDeclarationSyntax>();
        foreach (var (namespace_name, candidates) in Candidates.Select(kv => (kv.Key, kv.Value)))
        {
            var classes = new List<ClassDeclarationSyntax>();
            foreach (var cand in candidates)
            {
                var semanticModel = context.Compilation.GetSemanticModel(cand.methodDecl.SyntaxTree);
                // 1. generate the param preprocess
                var statements = BuildStatements(cand, target_interface);
                // 2. generate the IEvaluator Interface Impl
                var method = BuildMethod(cand, statements, target_interface);
                // 3. generate the classes
                var cls_name = cand.classDecl.Identifier.ValueText;
                var cls = ClassDeclaration(Identifier(cls_name))
                  .AddModifiers(
                      Token(SyntaxKind.PublicKeyword),
                      Token(SyntaxKind.PartialKeyword))
                  .AddBaseListTypes(cand.classDecl.BaseList.Types.ToArray())
                  .AddMembers(method);
                classes.Add(cls);
            }
            // 4. generate the namespaces
            namespaces.Add(NamespaceDeclaration(ParseName(namespace_name))
            .AddMembers(classes.ToArray()));
        }
        // 4. generate the file
        var compilationUnit = CompilationUnit().
                AddMembers(namespaces.ToArray()).
                AddUsings(
                  UsingDirective(ParseName("Nncase.IR")),
                  UsingDirective(ParseName("Nncase.IR.Math")),
                  UsingDirective(ParseName("Nncase.IR.NN")),
                  UsingDirective(ParseName("Nncase.IR.Tensors"))
                  ).
                NormalizeWhitespace();
        return compilationUnit;
    }
}
