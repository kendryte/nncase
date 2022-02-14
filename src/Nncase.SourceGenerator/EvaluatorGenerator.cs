using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;

namespace Nncase.SourceGenerator;


[Generator]
internal class EvaluatorGenerator : ISourceGenerator
{

    public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new IEvaluatorImplReceiver());

    public void Execute(GeneratorExecutionContext context)
    {
        if (context.SyntaxReceiver is not IEvaluatorImplReceiver evalReceiver)
            return;

        var namespaces = new List<NamespaceDeclarationSyntax>();
        foreach (var (namespace_name, candidates) in evalReceiver.Candidates.Select(kv => (kv.Key, kv.Value)))
        {
            var classes = new List<ClassDeclarationSyntax>();
            foreach (var cand in candidates)
            {
                var semanticModel = context.Compilation.GetSemanticModel(cand.methodDecl.SyntaxTree);
                // 1. generate the param preprocess
                var statements = BuildStatements(cand);
                // 2. generate the IEvaluator Interface Impl
                var method = BuildMethod(cand, statements);
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
        context.AddSource("Evaluator.Ops", SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText());
    }

    /// <summary>
    /// build the value convert expression like:
    /// var alpha = context.GetArgumentValueAsScalar<int>(celu, Celu.Alpha);
    /// </summary>
    /// <param name="cand"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    public List<StatementSyntax> BuildStatements(EvalCandidate cand)
    {
        var statements = new List<StatementSyntax>();
        TextInfo myTI = new CultureInfo("en-US", false).TextInfo;
        foreach (var param in cand.methodDecl.ParameterList.Parameters.Skip(2))
        {
            var paramName = param.Identifier.ValueText;
            string callMethod = param.Type switch
            {
                PredefinedTypeSyntax predefinedType=> $"GetArgumentValueAsScalar<{predefinedType.Keyword.ValueText}>",
                QualifiedNameSyntax qualified => IEvaluatorImplReceiver.GetFullName(qualified) switch
                {
                    var x when x.EndsWith("torch.Tensor") => "GetTorchArgumentValue",
                    var x when x.EndsWith("Tensorflow.Tensor") => "GetTFArgumentValue",
                    var x => throw new NotSupportedException(x)
                },
                _ => throw new NotSupportedException()
            };
            statements.Add(
              ParseStatement($"var {paramName} = context.{callMethod}({cand.OpParamIdentifier}, {cand.OpTypeName}.{myTI.ToTitleCase(paramName)});")
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
    MethodDeclarationSyntax BuildMethod(EvalCandidate cand, List<StatementSyntax> statements)
    {
        var method = MethodDeclaration(ParseTypeName("Const"), "Visit")
        .AddModifiers(Token(SyntaxKind.PublicKeyword))
        .AddParameterListParameters(
          Parameter(Identifier("context")).WithType(ParseTypeName("IEvaluateContext")),
          Parameter(Identifier(cand.OpParamIdentifier)).WithType(ParseTypeName(cand.OpTypeName)))
        .WithBody(Block(statements.ToArray()));
        return method;
    }
}
