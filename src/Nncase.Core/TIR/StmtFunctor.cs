// // Copyright (c) Canaan Inc. All rights reserved.
// // Licensed under the Apache license. See LICENSE file in the project root for full license information.

// using System;
// using System.Collections.Generic;
// using System.Linq;
// using System.Text;
// using System.Threading.Tasks;

// namespace Nncase.TIR
// {
//     /// <summary>
//     /// Expression functor.
//     /// </summary>
//     /// <typeparam name="TStmtResult">Expression visit result type.</typeparam>
//     /// <typeparam name="TExprResult">Expression visit result type.</typeparam>
//     /// <typeparam name="TTypeResult">Type visit result type.</typeparam>
//     public abstract class StmtFunctor<TStmtResult, TExprResult, TTypeResult> : Nncase.IR.ExprFunctor<TExprResult, TTypeResult>
//     {
//         /// <summary>
//         /// Visit expression.
//         /// </summary>
//         /// <param name="stmt">Expression.</param>
//         /// <returns>Result.</returns>
//         public virtual TStmtResult VisitStmt(Stmt stmt)
//         {
//             return stmt switch
//             {
//                 LetStmt let => VisitStmt(let),
//                 AttrStmt attrstmt => VisitStmt(attrstmt),
//                 IfThenElse ifthenelse => VisitStmt(ifthenelse),
//                 For @for => VisitStmt(@for),
//                 While @while => VisitStmt(@while),
//                 Allocate allocate => VisitStmt(allocate),
//                 Store store => VisitStmt(store),
//                 BufferStore bufferstore => VisitStmt(bufferstore),
//                 BufferRealize bufferrealize => VisitStmt(bufferrealize),
//                 AssertStmt assertstmt => VisitStmt(assertstmt),
//                 ProducerStore producerstore => VisitStmt(producerstore),
//                 Prefetch prefetch => VisitStmt(prefetch),
//                 SeqStmt seqstmt => VisitStmt(seqstmt),
//                 EvalExpr evalexpr => VisitStmt(evalexpr),
//                 Block block => VisitStmt(block),
//                 BlockRealize blockrealize => VisitStmt(blockrealize),
//                 _ => DefaultVisitStmt(stmt),
//             };
//         }
//         /// <summary>
//         /// visit LetStmt
//         /// </summary>
//         /// <param name="letstmt"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(LetStmt letstmt) => DefaultVisitStmt(letstmt);

//         /// <summary>
//         /// visit AttrStmt
//         /// </summary>
//         /// <param name="attrstmt"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(AttrStmt attrstmt) => DefaultVisitStmt(attrstmt);

//         /// <summary>
//         /// visit IfThenElse
//         /// </summary>
//         /// <param name="ifthenelse"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(IfThenElse ifthenelse) => DefaultVisitStmt(ifthenelse);

//         /// <summary>
//         /// visit For
//         /// </summary>
//         /// <param name="@for"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(For @for) => DefaultVisitStmt(@for);

//         /// <summary>
//         /// visit While
//         /// </summary>
//         /// <param name="@while"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(While @while) => DefaultVisitStmt(@while);

//         /// <summary>
//         /// visit Allocate
//         /// </summary>
//         /// <param name="allocate"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(Allocate allocate) => DefaultVisitStmt(allocate);

//         /// <summary>
//         /// visit Store
//         /// </summary>
//         /// <param name="store"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(Store store) => DefaultVisitStmt(store);

//         /// <summary>
//         /// visit BufferStore
//         /// </summary>
//         /// <param name="bufferstore"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(BufferStore bufferstore) => DefaultVisitStmt(bufferstore);

//         /// <summary>
//         /// visit BufferRealize
//         /// </summary>
//         /// <param name="bufferrealize"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(BufferRealize bufferrealize) => DefaultVisitStmt(bufferrealize);

//         /// <summary>
//         /// visit AssertStmt
//         /// </summary>
//         /// <param name="assertstmt"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(AssertStmt assertstmt) => DefaultVisitStmt(assertstmt);

//         /// <summary>
//         /// visit ProducerStore
//         /// </summary>
//         /// <param name="producerstore"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(ProducerStore producerstore) => DefaultVisitStmt(producerstore);

//         /// <summary>
//         /// visit Prefetch
//         /// </summary>
//         /// <param name="prefetch"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(Prefetch prefetch) => DefaultVisitStmt(prefetch);

//         /// <summary>
//         /// visit SeqStmt
//         /// </summary>
//         /// <param name="seqstmt"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(SeqStmt seqstmt) => DefaultVisitStmt(seqstmt);

//         /// <summary>
//         /// visit EvalExpr
//         /// </summary>
//         /// <param name="evalexpr"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(EvalExpr evalexpr) => DefaultVisitStmt(evalexpr);

//         /// <summary>
//         /// visit Block
//         /// </summary>
//         /// <param name="block"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(Block block) => DefaultVisitStmt(block);

//         /// <summary>
//         /// visit BlockRealize
//         /// </summary>
//         /// <param name="blockrealize"> inputs stmt </param>
//         /// <returns> returns. </returns>
//         public virtual TStmtResult VisitStmt(BlockRealize blockrealize) => DefaultVisitStmt(blockrealize);


//         /// <summary>
//         /// default visit
//         /// </summary>
//         /// <param name="stmt"></param>
//         /// <returns></returns>
//         /// <exception cref="NotImplementedException"></exception>
//         public virtual TStmtResult DefaultVisitStmt(Stmt stmt)
//         {
//             throw new NotImplementedException($"Unhandled visit routine for {stmt.GetType()}.");
//         }
//     }
// }
