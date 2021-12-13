using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using Microsoft.Build.Locator;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using static Microsoft.CodeAnalysis.CSharp.SyntaxFactory;
using Microsoft.CodeAnalysis.Formatting;
using Microsoft.CodeAnalysis.MSBuild;
using System.Text;

namespace PatternGenerator
{

    public class OpInfo
    {
        public string Name;
        public string PatternName;
        public List<ParameterSyntax> Params = new();

        public List<VariableDeclaratorSyntax> ParamInfos = new();
    }

    public class Receiver
    {
        private readonly Dictionary<string, List<OpInfo>> _candiateOps = new();

        private readonly Dictionary<string, List<MethodDeclarationSyntax>> _candiateFuncs = new();

        public Dictionary<string, List<OpInfo>> CandiateOps => _candiateOps;

        public Dictionary<string, List<MethodDeclarationSyntax>> CandiateFuncs => _candiateFuncs;

        private void CollectOpDef(SyntaxTree syntaxTree, string scope)
        {
            var root = syntaxTree.GetCompilationUnitRoot();
            var records = from record in root.DescendantNodes().OfType<RecordDeclarationSyntax>()
                          select record;
            foreach (var record in records)
            {   /// NOTE the record must has ()
                if (record.ParameterList is null)
                    continue;
                var op = new OpInfo()
                {
                    Name = record.Identifier.ValueText,
                    PatternName = record.Identifier.ValueText + "Pattern",
                    Params = record.ParameterList.Parameters.ToList(),
                    ParamInfos = (from declar in record.DescendantNodes().OfType<VariableDeclaratorSyntax>()
                                  where (declar is VariableDeclaratorSyntax { Parent: VariableDeclarationSyntax { Parent: FieldDeclarationSyntax { } field } parent })
                                  select declar).ToList()
                };
                Console.WriteLine($"Add Record {op.Name}");
                if (!_candiateOps.TryGetValue(scope, out var list))
                    _candiateOps.Add(scope, new());
                _candiateOps[scope].Add(op);
            }
        }

        private void CollectOpFunc(SyntaxTree syntaxTree, string scope)
        {
            var root = syntaxTree.GetCompilationUnitRoot();
            var funcs = from func in root.DescendantNodes().OfType<MethodDeclarationSyntax>()
                        select func;
            foreach (var func in funcs)
            {
                Console.WriteLine($"Add Functional {func.Identifier.ValueText}");
                if (!_candiateFuncs.TryGetValue(scope, out var list))
                    _candiateFuncs.Add(scope, new());
                _candiateFuncs[scope].Add(func);
            }
        }

        public async Task VisitProject(Solution solution, Project project)
        {

            var compilation = await project.GetCompilationAsync();
            foreach (DocumentId documentId in project.DocumentIds)
            {
                // Look up the snapshot for the original document in the latest forked solution.
                Document document = solution.GetDocument(documentId);
                Console.WriteLine("  " + document.FilePath);
                if (!(document.Folders.Count >= 2 && document.Folders[document.Folders.Count - 2] == "IR"))
                    continue;
                var tree = await document.GetSyntaxTreeAsync();
                if (document.FilePath.Contains("Functional.cs"))
                {
                    CollectOpFunc(tree, document.Folders.Last());
                }
                else
                {
                    CollectOpDef(tree, document.Folders.Last());
                }
            }
        }
    }

    public static class Generator
    {

        /// <summary>
        /// 从
        /// </summary>
        /// <param name="receiver"></param>
        /// <param name="filePath"></param>
        private static void GenerateWrappers(Receiver receiver, string filePath)
        {
            var baseType = ParseTypeName("PatternWrapper");
            var paramInfoType = ParseTypeName("ParameterInfo");
            var wrappers = new List<RecordDeclarationSyntax>();
            var namespcaes = new List<NamespaceDeclarationSyntax>();
            foreach (var (scope, ops) in receiver.CandiateOps)
            {
                foreach (var op in ops)
                {
                    var name = $"{op.Name}Wrapper";

                    var getInputMembers = op.ParamInfos.SelectMany(info =>
                      {
                          var pname = info.Identifier.ValueText;
                          return new[]{
                      ParseMemberDeclaration($"public ExprPattern {pname}Pat() => Pattern[{op.Name}.{pname}];"),
                      ParseMemberDeclaration($"public T {pname}Pat<T>() where T : ExprPattern => (T){pname}Pat();"),
                      ParseMemberDeclaration($"public Expr {pname}() => GetCast<Expr>({pname}Pat());"),
                      ParseMemberDeclaration($"public T {pname}<T>() where T : Expr => GetCast<T>({pname}Pat());")
                          };
                      });

                    var getEnumMembers = (from param in op.Params
                                          let pname = param.Identifier.ValueText
                                          let ptype = param.Type
                                          select ParseMemberDeclaration($"public {ptype} {pname} => (({op.Name})GetCast<Call>(this).Target).{pname};"));

                    var wrapper = RecordDeclaration(Token(SyntaxKind.RecordKeyword), name).
                    AddModifiers(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.SealedKeyword)).
                    AddParameterListParameters(
                      Parameter(Identifier("Pattern")).
                      WithType(ParseTypeName("CallPattern"))
                    ).
                    AddBaseListTypes(SimpleBaseType(baseType)).
                    WithOpenBraceToken(Token(SyntaxKind.OpenBraceToken)).
                    AddMembers(getInputMembers.ToArray()).
                    AddMembers(getEnumMembers.ToArray()).
                    AddMembers(
                      ParseMemberDeclaration($"public static implicit operator CallPattern({name} warper) => warper.Pattern;")
                    ).
                    WithCloseBraceToken(Token(SyntaxKind.CloseBraceToken));

                    wrappers.Add(wrapper);
                }
                var @namespace = NamespaceDeclaration(ParseName($"Nncase.Pattern.{scope}")).AddMembers(wrappers.ToArray());
                namespcaes.Add(@namespace);
                wrappers.Clear();
            }

            var compilationUnit = CompilationUnit().
                AddMembers(namespcaes.ToArray()).
                AddUsings(
                  UsingDirective(ParseName("System")),
                  UsingDirective(ParseName("System.Collections.Generic")),
                  UsingDirective(ParseName("System.Collections.Immutable")),
                  UsingDirective(ParseName("System.Linq")),
                  UsingDirective(ParseName("System.Text")),
                  UsingDirective(ParseName("System.Threading.Tasks")),
                  UsingDirective(ParseName("Nncase.IR")),
                  UsingDirective(ParseName("Nncase.IR.Math")),
                  UsingDirective(ParseName("Nncase.IR.NN")),
                  UsingDirective(ParseName("Nncase.IR.Tensors"))
                  ).
                NormalizeWhitespace();

            var sourceText = SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText();
            var file = File.Open(Path.Combine(filePath, "Generated.OpsWrapper.cs"), FileMode.Create);
            var writer = new StreamWriter(file);
            writer.Write(sourceText);
            writer.Close();
        }
        private static void GenerateDefs(Receiver receiver, string filePath)
        {

            var baseType = SimpleBaseType(ParseTypeName("OpPattern"));
            var patterns = new List<RecordDeclarationSyntax>();
            var namespcaes = new List<NamespaceDeclarationSyntax>();
            foreach (var (scope, ops) in receiver.CandiateOps)
            {
                foreach (var op in ops)
                {
                    var name = $"{op.Name}Pattern";

                    var members = (from param in op.Params
                                   let pname = param.Identifier.ValueText
                                   let ptype = param.Type
                                   select ParseMemberDeclaration($"public {name}({ptype} {pname}) : this(({op.Name} x) => {pname} == x.{pname}) {{ this.{pname} = {pname}; }}")
                    );

                    var fields = (from param in op.Params
                                  let pname = param.Identifier.ValueText
                                  let ptype = param.Type
                                  select ParseMemberDeclaration($"public {ptype}? {pname} = null;")
                    );

                    string init_fields = String.Join("\n",
                      (from param in op.Params
                       let pname = param.Identifier.ValueText
                       let ptype = param.Type
                       select ($"this.{pname} = {op.Name.ToLower()}.{pname};")
                    ));

                    var pattern = RecordDeclaration(Token(SyntaxKind.RecordKeyword), name).
                    AddModifiers(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.SealedKeyword)).
                    AddParameterListParameters(
                      Parameter(Identifier("Cond")).
                      WithType(ParseTypeName($"Func<{op.Name}, bool>"))
                    ).
                    AddBaseListTypes(baseType).
                    WithOpenBraceToken(Token(SyntaxKind.OpenBraceToken)).
                    AddMembers(
                      ParseMemberDeclaration($"public {name}({op.Name} {op.Name.ToLower()}) : this(x => x == {op.Name.ToLower()}) {{ {init_fields} }}"),
                      ParseMemberDeclaration($"public bool MatchLeaf({op.Name} {op.Name.ToLower()}) => Cond({op.Name.ToLower()}) && MatchCheckedType({op.Name.ToLower()});"),
                      ParseMemberDeclaration($"public {name}() : this(({op.Name} x) => true) {{ }}")
                    ).
                    AddMembers(fields.ToArray()).
                    AddMembers(members.ToArray()).
                    WithCloseBraceToken(Token(SyntaxKind.CloseBraceToken));

                    patterns.Add(pattern);
                }
                var @namespace = NamespaceDeclaration(ParseName($"Nncase.Pattern.{scope}")).AddMembers(patterns.ToArray());
                namespcaes.Add(@namespace);
                patterns.Clear();
            }

            var compilationUnit = CompilationUnit().
                AddMembers(namespcaes.ToArray()).
                AddUsings(
                  UsingDirective(ParseName("System")),
                  UsingDirective(ParseName("System.Collections.Generic")),
                  UsingDirective(ParseName("System.Collections.Immutable")),
                  UsingDirective(ParseName("System.Linq")),
                  UsingDirective(ParseName("System.Text")),
                  UsingDirective(ParseName("System.Threading.Tasks")),
                  UsingDirective(ParseName("Nncase.IR")),
                  UsingDirective(ParseName("Nncase.IR.Math")),
                  UsingDirective(ParseName("Nncase.IR.NN")),
                  UsingDirective(ParseName("Nncase.IR.Tensors"))
                  ).
                NormalizeWhitespace();

            var sourceText = SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText();
            var file = File.Open(Path.Combine(filePath, "Generated.OpsPatten.cs"), FileMode.Create);
            var writer = new StreamWriter(file);
            writer.Write(sourceText);
            writer.Close();
        }

        public static void GenerateOpVisits(Receiver receiver, string filePath)
        {
            var baseType = SimpleBaseType(ParseTypeName("ExprPattern"));
            var visitArms = new List<SwitchExpressionArmSyntax>();
            var castArms = new List<SwitchExpressionArmSyntax>(); /* cast the op to OpPattern visit */
            foreach (var (scope, ops) in receiver.CandiateOps)
            {
                foreach (var op in ops)
                {
                    var lhsDeclar = DeclarationPattern(ParseTypeName(op.PatternName), SingleVariableDesignation(Identifier(op.PatternName.ToLower())));
                    var rhsDeclar = DeclarationPattern(ParseTypeName(op.Name), SingleVariableDesignation(Identifier(op.Name.ToLower())));
                    var combineDeclar = RecursivePattern().AddPositionalPatternClauseSubpatterns(Subpattern(lhsDeclar), Subpattern(rhsDeclar));
                    var invocation = ParseExpression($"{op.PatternName.ToLower()}.MatchLeaf({op.Name.ToLower()})");
                    visitArms.Add(SwitchExpressionArm(combineDeclar, invocation));

                    castArms.Add(SwitchExpressionArm(rhsDeclar, ParseExpression($"new {op.PatternName}({op.Name.ToLower()})")));
                }
            }
            visitArms.Add(SwitchExpressionArm(
                     RecursivePattern().
                     AddPositionalPatternClauseSubpatterns(
                       Subpattern(DiscardPattern()),
                       Subpattern(DiscardPattern())),
                     ParseExpression("false")));

            castArms.Add(SwitchExpressionArm(DiscardPattern(),
                     ParseExpression(@"throw new NotImplementedException($""Can't Convert OP {op.GetType().Name} To ExprPattern"")")));

            var visitSwitch = SwitchExpression(
                  ParseExpression("(this, op)"),
                  SeparatedList(visitArms));

            var matchMethod = MethodDeclaration(ParseTypeName("bool"), "MatchLeaf")
                .AddModifiers(Token(SyntaxKind.PublicKeyword))
                .AddParameterListParameters(
                    Parameter(Identifier("op")).WithType(ParseTypeName("Op")))
                .WithLeadingTrivia(LineFeed)
                .WithExpressionBody(ArrowExpressionClause(visitSwitch))
                .WithSemicolonToken(Token(SyntaxKind.SemicolonToken));

            /* build cast method */
            var castSwitch = SwitchExpression(
                  ParseExpression("op"),
                  SeparatedList(castArms));

            var castMethod = MethodDeclaration(ParseTypeName("ExprPattern"), "CastToPattern")
                .AddModifiers(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.StaticKeyword))
                .AddParameterListParameters(
                    Parameter(Identifier("op")).WithType(ParseTypeName("Op")))
                .WithLeadingTrivia(LineFeed)
                .WithExpressionBody(ArrowExpressionClause(castSwitch))
                .WithSemicolonToken(Token(SyntaxKind.SemicolonToken));

            var @record = RecordDeclaration(Token(SyntaxKind.RecordKeyword), "OpPattern")
                          .AddModifiers(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.AbstractKeyword), Token(SyntaxKind.PartialKeyword))
                          .AddBaseListTypes(SimpleBaseType(ParseTypeName("ExprPattern")))
                          .WithOpenBraceToken(Token(SyntaxKind.OpenBraceToken))
                          .AddMembers(matchMethod)
                          .AddMembers(castMethod)
                          .WithCloseBraceToken(Token(SyntaxKind.CloseBraceToken));

            var @namespace = NamespaceDeclaration(ParseName("Nncase.Pattern"))
                .AddMembers(@record);

            var compilationUnit = CompilationUnit()
                .AddMembers(@namespace)
                .AddUsings(
            UsingDirective(ParseName("System")),
            UsingDirective(ParseName("System.Collections.Generic")),
            UsingDirective(ParseName("System.Collections.Immutable")),
            UsingDirective(ParseName("System.Linq")),
            UsingDirective(ParseName("Nncase.IR")),
            UsingDirective(ParseName("Nncase.IR.NN")),
            UsingDirective(ParseName("Nncase.IR.Math")),
            UsingDirective(ParseName("Nncase.IR.Tensors")),
            UsingDirective(ParseName("Nncase.Pattern.NN")),
            UsingDirective(ParseName("Nncase.Pattern.Math")),
            UsingDirective(ParseName("Nncase.Pattern.Tensors")))
                .NormalizeWhitespace();
            var syntaxTree = SyntaxTree(compilationUnit, encoding: Encoding.UTF8);
            var file = File.Open(Path.Combine(filePath, "Generated.OpPattern.cs"), FileMode.Create);
            var writer = new StreamWriter(file);
            writer.Write(syntaxTree.GetText());
            writer.Close();
        }

        private static readonly Dictionary<string, string> returnDict = new()
        {
            { "Abs", "Unary" },
            { "Ceil", "Unary" },
            { "Cos", "Unary" },
            { "Exp", "Unary" },
            { "Floor", "Unary" },
            { "Log", "Unary" },
            { "Neg", "Unary" },
            { "Round", "Unary" },
            { "Rsqrt", "Unary" },
            { "Sin", "Unary" },
            { "Sqrt", "Unary" },
            { "Square", "Unary" },
            { "Tanh", "Unary" },
            { "BitwiseNot", "Unary" },
            { "LogicalNot", "Unary" },
            { "Add", "Binary" },
            { "Sub", "Binary" },
            { "Mul", "Binary" },
            { "Div", "Binary" },
            { "Mod", "Binary" },
            { "Min", "Binary" },
            { "Max", "Binary" },
            { "Pow", "Binary" },
            { "BitwiseAnd", "Binary" },
            { "BitwiseOr", "Binary" },
            { "BitwiseXor", "Binary" },
            { "LogicalAnd", "Binary" },
            { "LogicalOr", "Binary" },
            { "LogicalXor", "Binary" },
            { "LeftShift", "Binary" },
            { "RightShift", "Binary" },
            { "Equal", "Compare" },
            { "NotEqual", "Compare" },
            { "LessThan", "Compare" },
            { "LessEqual", "Compare" },
            { "GreaterEqual", "Compare" },
            { "GreaterThan", "Compare" },
            { "FloorDiv", "Unary" },
            { "FloorMod", "Binary" },
            { "ReduceMean", "Reduce" },
            { "ReduceMin", "Reduce" },
            { "ReduceMax", "Reduce" },
            { "ReduceSum", "Reduce" }
        };

        public static void GenerateFuncs(Receiver receiver, string filePath)
        {
            var functions = new List<MemberDeclarationSyntax>();
            var classes = new List<ClassDeclarationSyntax>();
            var regex = new Regex(@"(new CallPattern(.*)(?=;))", RegexOptions.Compiled | RegexOptions.IgnoreCase);
            var dict = new Dictionary<string, string>() {
              {"Expr" ,"ExprPattern"},
              {"Call" ,"CallPattern"},
              {"Tuple" ,"TuplePattern"},
            };
            foreach (var (_, ops) in receiver.CandiateOps)
            {
                foreach (var op in ops)
                {
                    dict.TryAdd($"new {op.Name}(", $"new {op.PatternName}(");
                }
            }

            foreach (var (scope, funcs) in receiver.CandiateFuncs)
            {
                foreach (var func in funcs)
                {
                    string newfunc = func.GetText().ToString();
                    var funcName = func.Identifier.ValueText;
                    foreach (var (oldname, newname) in dict)
                    {
                        newfunc = newfunc.Replace(oldname, newname);
                    }
                    if (!returnDict.TryGetValue(funcName, out var newReturnName))
                    {
                        newReturnName = funcName;
                    }
                    newfunc = newfunc.Replace("public static CallPattern", $"public static {newReturnName}Wrapper");
                    newfunc = regex.Replace(newfunc, $"new {funcName}Wrapper($&)");
                    functions.Add(ParseMemberDeclaration(newfunc));
                }
                var @class = ClassDeclaration(Identifier(scope)).
                         AddModifiers(Token(SyntaxKind.PublicKeyword), Token(SyntaxKind.StaticKeyword), Token(SyntaxKind.PartialKeyword)).
                         AddMembers(functions.ToArray());
                functions.Clear();
                classes.Add(@class);
            }
            var @namespace = NamespaceDeclaration(ParseName($"Nncase.Pattern.F")).AddMembers(classes.ToArray());
            var compilationUnit = CompilationUnit().
                AddMembers(@namespace).
                AddUsings(
                  UsingDirective(ParseName("System.Collections.Generic")),
                  UsingDirective(ParseName("System.Linq")),
                  UsingDirective(ParseName("System.Text")),
                  UsingDirective(ParseName("System.Threading.Tasks")),
                  UsingDirective(ParseName("Nncase.Pattern.Math")),
                  UsingDirective(ParseName("Nncase.Pattern.NN")),
                  UsingDirective(ParseName("Nncase.Pattern.Tensors")),
                  UsingDirective(ParseName("Nncase.IR")),
                  UsingDirective(ParseName("Nncase.IR.Math")),
                  UsingDirective(ParseName("Nncase.IR.NN")),
                  UsingDirective(ParseName("Nncase.IR.Tensors"))
                  ).
                NormalizeWhitespace();

            var sourceText = SyntaxTree(compilationUnit, encoding: Encoding.UTF8).GetText();
            var file = File.Open(Path.Combine(filePath, "Generated.Functional.cs"), FileMode.Create);
            var writer = new StreamWriter(file);
            writer.Write(sourceText);
            writer.Close();
        }

        public static void Generate(Receiver receiver, string filePath)
        {
            GenerateWrappers(receiver, filePath);
            GenerateDefs(receiver, filePath);
            GenerateFuncs(receiver, filePath);
            GenerateOpVisits(receiver, filePath);
        }
    }

    class Program
    {
        static async Task Main(string[] args)
        {
            var nncaseRoot = new DirectoryInfo("../../").FullName;
            var patternRoot = Path.Combine(nncaseRoot, "src", "Nncase.Pattern");

            // Console.WriteLine("Hello World!");
            // Locate and register the default instance of MSBuild installed on this machine.
            MSBuildLocator.RegisterDefaults();

            // The test solution is copied to the output directory when you build this sample.
            MSBuildWorkspace workspace = MSBuildWorkspace.Create();

            // Open the solution within the workspace.
            Solution solution = await workspace.OpenSolutionAsync(Path.Combine(nncaseRoot, "nncase.sln"));

            var reciever = new Receiver();
            foreach (var projectId in solution.ProjectIds)
            {
                // Look up the snapshot for the original project in the latest forked solution.
                Project project = solution.GetProject(projectId);
                Console.WriteLine(project.Name);
                var compilation = await project.GetCompilationAsync();
                switch (project.Name)
                {
                    case "Nncase.Core":
                        await reciever.VisitProject(solution, project);
                        break;
                    case "Nncase.IR":
                        await reciever.VisitProject(solution, project);
                        break;
                    default:
                        break;
                }
            }

            Generator.Generate(reciever, patternRoot);
        }
    }
}
