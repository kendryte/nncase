using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;

namespace Nncase.SourceGenerator.Evaluator;



[Generator]
internal class EvaluatorGenerator : ISourceGenerator
{

    GeneratorExecutionContext? Context = null;
    EvaluatorImplReceiver Receiver;

    public void Initialize(GeneratorInitializationContext context) => context.RegisterForSyntaxNotifications(() => new EvaluatorImplReceiver());

    public void Execute(GeneratorExecutionContext context)
    {
        if (context.SyntaxContextReceiver is not EvaluatorImplReceiver evalReceiver)
            return;
        Receiver = evalReceiver;
        Context ??= context;
        if (evalReceiver.Diagnostics.Any())
        {
            foreach (var diag in evalReceiver.Diagnostics)
            {
                context.ReportDiagnostic(diag);
            }
        }
        if (evalReceiver.EvalCandidates.Any())
        {
            var eval_compilationunit = BuildFile(context, evalReceiver.EvalCandidates);
            context.AddSource("Ops.Evaluator", SyntaxTree(eval_compilationunit, encoding: Encoding.UTF8).GetText());
        }

        if (evalReceiver.TypeInferCandidates.Any())
        {
            var typeinfer_compilationunit = BuildFile(context, evalReceiver.TypeInferCandidates);
            context.AddSource("Ops.TypeInferencer", SyntaxTree(typeinfer_compilationunit, encoding: Encoding.UTF8).GetText());
        }
    }

    /// <summary>
    /// build the value convert expression like:
    /// var alpha = context.GetArgumentValueAsScalar<int>(celu, Celu.Alpha);
    /// </summary>
    /// <param name="cand"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    List<StatementSyntax> BuildStatements(GenerateCandidate cand)
    {
        var (return_type_name, context_type_name) = cand.Target.GetKindInfo();
        var statementSyntaxes = new List<StatementSyntax>();
        foreach (var Parameter in cand.Method.Parameters)
        {
            // if (!cand.Op.MemberNames.Any(name => name == Parameter.Name))
            //     Context.Value.ReportDiagnostic(Diagnostic.Create(RecriverUtil.MethodParamError, Location.None, cand.Class.Name, string.Join(", ", cand.Method.Parameters.Select(p => p.Name)), cand.Op.Name));
            var paramType = Parameter.Type;
            if ((cand.Target == InterfaceKind.IEvaluator && paramType.Equals(Receiver.IEvaluateContextSymobl))
              || (cand.Target == InterfaceKind.ITypeInferencer && paramType.Equals(Receiver.ITypeInferenceContext)))
            {
                if (Parameter.Name != "context")
                    statementSyntaxes.Add(ParseStatement($"var {Parameter.Name} = context;"));
                continue;
            }
            if (paramType.Equals(cand.Op))
            {
                if (Parameter.Name != "target")
                    statementSyntaxes.Add(ParseStatement($"var {Parameter.Name} = target;"));
                continue;
            }
            string callMethod = cand.Target switch
            {
                InterfaceKind.IEvaluator => paramType switch
                {
                    { IsReferenceType: true } x when x.IsInheritFrom(Receiver.ExprSymobl) => $"GetArgumentExpr<{paramType.ToDisplayString()}>",
                    { IsReferenceType: true } x when x.ToDisplayString().EndsWith("torch.Tensor") => "GetTorchArgumentValue",
                    { IsReferenceType: true } x when x.ToDisplayString().EndsWith("Tensorflow.Tensor") => "GetTFArgumentValue",
                    { IsUnmanagedType: true, IsValueType: true } x => $"GetArgumentValueAsScalar<{paramType.ToDisplayString()}>",
                    _ => throw new NotSupportedException($"Convert {paramType.ToDisplayString()} For IEvaluator Impl!")
                },
                InterfaceKind.ITypeInferencer => paramType switch
                {
                    { IsReferenceType: true } x when x.IsInheritFrom(Receiver.IRTypeSymobl) => $"CheckArgumentType<{x}>"
                },
                _ => throw new NotSupportedException($"{paramType.ToDisplayString()} with {cand.Target}!")
            };

            statementSyntaxes.Add(ParseStatement($"var {Parameter.Name} = context.{callMethod}(target, {cand.Op.ToDisplayString()}.{Parameter.Name});"));
        }
        var visitMethod = cand.Method.ReturnType.BuildReturnWrapper(cand.Target, $"Visit({string.Join(",", cand.Method.Parameters.Select(p => p.Name))})");
        statementSyntaxes.Add(ParseStatement($"return {visitMethod};"));
        return statementSyntaxes;
    }

    /// <summary>
    /// build the whole call method like:
    /// <code>
    /// public IValue Visit(IEvaluateContext context, Celu celu)
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
    MethodDeclarationSyntax BuildMethod(GenerateCandidate cand, List<StatementSyntax> statements)
    {
        var (return_type_name, context_type_name) = cand.Target.GetKindInfo();
        var method = MethodDeclaration(ParseTypeName(return_type_name), "Visit")
        .AddModifiers(Token(SyntaxKind.PublicKeyword))
        .AddParameterListParameters(
          Parameter(Identifier("context")).WithType(ParseTypeName(context_type_name)),
          Parameter(Identifier("target")).WithType(ParseTypeName(cand.Op.ToDisplayString())))
        .WithBody(Block(statements.ToArray()));
        return method;
    }

    CompilationUnitSyntax BuildFile(GeneratorExecutionContext context, List<GenerateCandidate> Candidates)
    {
        List<NamespaceDeclarationSyntax> namespaceDeclarations = new();
        var NamespaceCandidates = Candidates.GroupBy(keySelector: can => can.Class.ContainingNamespace).ToDictionary(g => g.Key, g => g.ToList());
        foreach (var (Namespace, candidates) in NamespaceCandidates.Select(kv => (kv.Key, kv.Value)))
        {
            List<ClassDeclarationSyntax> classDeclarations = new();
            foreach (var cand in candidates)
            {
                // 1. generate the param preprocess
                var statementSyntaxes = BuildStatements(cand);
                // 2. generate the IEvaluator Interface Impl
                var methodDeclarations = BuildMethod(cand, statementSyntaxes);
                // 3. generate the classes
                var cls_name = cand.Class.Name;
                var cls = ClassDeclaration(Identifier(cls_name))
                  .AddModifiers(
                      Token(SyntaxKind.PublicKeyword),
                      Token(SyntaxKind.PartialKeyword))
                  .AddBaseListTypes(cand.Class.Interfaces.Select(i => SimpleBaseType(ParseTypeName(i.ToDisplayString()))).ToArray())
                  .AddMembers(methodDeclarations);
                classDeclarations.Add(cls);
            }
            // 4. generate the namespaces
            namespaceDeclarations.Add(NamespaceDeclaration(ParseName(Namespace.ToDisplayString()))
            .AddMembers(classDeclarations.ToArray()));
        }

        // 4. generate the file
        var compilationUnit = CompilationUnit().
                AddMembers(namespaceDeclarations.ToArray()).
                NormalizeWhitespace();
        return compilationUnit;
    }
}
