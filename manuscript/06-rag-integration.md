# RAGとの統合：検索と生成とエージェント構成

## RAGの基本概念

RAGはドキュメントから知識を抽出してLLMに適切に供給する仕組みです。
Retrieval-Augmented Generation（検索拡張生成）は、外部知識ベースから関連情報を検索し、それをLLMのコンテキストに含めて回答を生成する手法です。
LLMの学習データには含まれない最新情報や、企業固有のドメイン知識をLLMに提供できます。

RAGは以下の3つのステップで構成されます。

### 1. ドキュメントの準備と埋め込み生成

ドキュメントを小さなチャンク（塊）に分割し、各チャンクをベクトル化します。
ベクトル化には OpenAI の Embeddings API や、オープンソースの sentence-transformers などを使用します。
生成されたベクトルは、ベクトルデータベース（Pinecone、Weaviate、FAISS など）に格納されます。

### 2. ユーザークエリの検索

ユーザーの質問もベクトル化し、ベクトルデータベース内で類似度検索を行います。
コサイン類似度などの指標を用いて、質問に最も関連性の高いドキュメントチャンクを取得します。
検索結果は通常、上位 k 件（例：k=3 や k=5）を取得します。

### 3. LLMによる回答生成

検索で取得したドキュメントチャンクを、LLM のプロンプトに含めます。
LLM は提供された文脈を参照して、ユーザーの質問に回答します。
これにより、LLM が学習していない情報に基づいた正確な回答が可能になります。

## RAGとエージェントの関係

RAGはエージェントのツールとして機能し、検索と生成と意思決定を結び付けます。
エージェントは RAG をツールの一つとして利用し、必要に応じて知識ベースを検索します。
例えば、ユーザーが「製品 A の仕様を教えて」と質問した場合、エージェントは RAG ツールを呼び出し、製品ドキュメントから関連情報を取得します。

RAG をエージェントに統合することで、以下のメリットがあります。

- **知識の鮮度**: 外部ドキュメントを更新するだけで、エージェントの知識を最新に保てる
- **ドメイン専門性**: 企業固有の文書を活用して、専門的な質問に答えられる
- **説明可能性**: 回答の根拠となったドキュメントを提示できる
- **コスト削減**: LLM の学習データに含めるのではなく、検索で補完することでコストを抑える

## ベクトル検索の仕組み

ベクトル検索と類似度計算とチャンク分割の基本概念を整理します。

### 埋め込みベクトル（Embeddings）

埋め込みベクトルは、テキストを多次元空間の点として表現したものです。
意味的に類似したテキストは、ベクトル空間上で近い位置に配置されます。
例えば、「犬」と「猫」のベクトルは、「犬」と「自動車」のベクトルよりも近くなります。

OpenAI の `text-embedding-3-small` や `text-embedding-3-large` は、高品質な埋め込みを生成します。
これらのモデルは、多言語対応しており、日本語のテキストも適切にベクトル化できます。

### 類似度計算

ベクトル間の類似度は、コサイン類似度やユークリッド距離で計算されます。
コサイン類似度は、ベクトルの角度に基づいて類似度を測定し、-1 から 1 の値を取ります。
1 に近いほど類似しており、-1 に近いほど異なります。

### チャンク分割戦略

ドキュメントを適切なサイズのチャンクに分割することが重要です。
チャンクが大きすぎると、無関係な情報が含まれて検索精度が下がります。
チャンクが小さすぎると、文脈が失われて理解が困難になります。

一般的なチャンク分割の方針：

- **固定長分割**: 文字数やトークン数で機械的に分割（簡単だが文脈が切れやすい）
- **段落分割**: 段落単位で分割（文脈を保ちやすい）
- **セマンティック分割**: 意味的なまとまりで分割（高度だが効果的）
- **オーバーラップ**: チャンク間で一部重複させる（文脈の連続性を保つ）

## LangChainでのRAG構成パターン

LangChainとLangGraphの双方で成立するRAG構成パターンを紹介します。

### DocumentLoader によるドキュメント読み込み

LangChain は多様な DocumentLoader を提供しています。

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

# テキストファイルの読み込み
text_loader = TextLoader("documents/product_manual.txt")
documents = text_loader.load()

# PDFファイルの読み込み
pdf_loader = PyPDFLoader("documents/specification.pdf")
pdf_documents = pdf_loader.load()

# Webページの読み込み
web_loader = WebBaseLoader("https://example.com/docs")
web_documents = web_loader.load()
```

### TextSplitter によるチャンク分割

ドキュメントを適切なサイズに分割します。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # チャンクの最大文字数
    chunk_overlap=200,  # チャンク間のオーバーラップ
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # 分割の優先順位
)

chunks = text_splitter.split_documents(documents)
print(f"分割されたチャンク数: {len(chunks)}")
```

### VectorStore によるベクトル化と格納

ベクトルストアにチャンクを格納します。

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 埋め込みモデルの初期化
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# FAISSベクトルストアの構築
vectorstore = FAISS.from_documents(chunks, embeddings)

# ベクトルストアの保存（再利用可能）
vectorstore.save_local("vectorstore/product_docs")

# 保存したベクトルストアの読み込み
# vectorstore = FAISS.load_local("vectorstore/product_docs", embeddings)
```

## 実装レベル1：基本的なRAG検索と生成

実装レベル1ではDocumentLoaderとVectorStoreを使った簡易検索と生成を示します。

### シンプルなRAGシステムの構築

ドキュメントを読み込み、ベクトル化し、検索して回答を生成する最小構成を実装します。

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. ドキュメントの読み込み
loader = TextLoader("documents/company_faq.txt")
documents = loader.load()

# 2. チャンク分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 3. ベクトルストアの構築
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. プロンプトテンプレート
template = """以下の文脈を参考にして、質問に答えてください。
文脈に関連する情報がない場合は、「情報がありません」と答えてください。

文脈:
{context}

質問: {question}

回答:"""

prompt = ChatPromptTemplate.from_template(template)

# 5. LLMの初期化
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 6. RAGチェーンの構築（LCEL使用）
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 実行
question = "営業時間を教えてください"
answer = rag_chain.invoke(question)
print(f"質問: {question}")
print(f"回答: {answer}")
```

このコードでは、ドキュメントを読み込み、検索可能な形式に変換し、質問に対して関連情報を検索して回答を生成します。

### 検索結果の確認

検索がどのドキュメントを取得したか確認することが重要です。

```python
# 検索結果を取得
retrieved_docs = retriever.invoke(question)

print(f"\n取得されたドキュメント数: {len(retrieved_docs)}")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- ドキュメント {i+1} ---")
    print(doc.page_content)
    print(f"メタデータ: {doc.metadata}")
```

これにより、どの文脈が LLM に渡されたかを確認できます。

## 実装レベル2：エージェント統合とセッション管理

実装レベル2では全文検索と生成とセッション状態管理とエージェント化を行います。

### RAGをツールとして定義

RAG をエージェントのツールとして組み込みます。

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# サンプルドキュメントを準備
documents = [
    Document(page_content="当社の営業時間は平日9時から18時までです。土日祝日は休業しております。", metadata={"source": "faq"}),
    Document(page_content="製品Aの保証期間は購入日から1年間です。故障時は無償修理いたします。", metadata={"source": "warranty"}),
    Document(page_content="配送は通常、注文確定後3営業日以内に行います。北海道と沖縄は5営業日かかります。", metadata={"source": "shipping"}),
]

# ベクトルストア構築
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

@tool
def search_company_docs(query: str) -> str:
    """社内ドキュメントから関連情報を検索します。営業時間、製品情報、配送情報などを調べる際に使用してください。"""
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        return "関連する情報が見つかりませんでした。"

    # 検索結果を整形
    content = "\n\n".join([f"[情報源: {doc.metadata['source']}]\n{doc.page_content}" for doc in results])
    return content

# エージェントの構築
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, [search_company_docs])

# エージェントの実行
response = agent.invoke({
    "messages": [HumanMessage(content="製品Aの保証について教えてください")]
})

print(response["messages"][-1].content)
```

エージェントは質問内容を理解し、適切なツール（RAG検索）を呼び出して回答を生成します。

### 複数のRAGツールを組み合わせる

異なる知識ベースに対する複数の RAG ツールを定義できます。

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 製品情報ドキュメント
product_docs = [
    Document(page_content="製品Aは高性能プロセッサを搭載し、軽量設計です。", metadata={"type": "product"}),
    Document(page_content="製品Bは大容量バッテリーで長時間駆動が可能です。", metadata={"type": "product"}),
]

# 技術サポートドキュメント
support_docs = [
    Document(page_content="エラーコード101は接続エラーです。ネットワーク設定を確認してください。", metadata={"type": "support"}),
    Document(page_content="再起動で多くの問題が解決します。電源ボタンを10秒長押ししてください。", metadata={"type": "support"}),
]

embeddings = OpenAIEmbeddings()
product_store = FAISS.from_documents(product_docs, embeddings)
support_store = FAISS.from_documents(support_docs, embeddings)

@tool
def search_product_info(query: str) -> str:
    """製品の仕様や特徴を検索します。"""
    results = product_store.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results])

@tool
def search_technical_support(query: str) -> str:
    """技術サポート情報を検索します。エラーコードやトラブルシューティング情報を調べます。"""
    results = support_store.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results])

# エージェントに両方のツールを渡す
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, [search_product_info, search_technical_support])

# エージェントの実行
response = agent.invoke({
    "messages": [HumanMessage(content="エラーコード101が表示されました。どうすればいいですか？")]
})

print(response["messages"][-1].content)
```

エージェントは質問の内容に応じて、適切な知識ベースを選択して検索します。

## 実装レベル3：LangGraphによる高度なRAGワークフロー

LangGraph を使って、より複雑な RAG ワークフローを構築します。

### クエリ改善と再検索を含むRAGフロー

ユーザーの質問が曖昧な場合、LLM を使ってクエリを改善してから検索します。

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

class RAGState(TypedDict):
    question: str
    refined_query: str
    retrieved_docs: List[str]
    answer: str

# サンプルドキュメント
documents = [
    Document(page_content="LangGraphは複雑なワークフローを構築するためのライブラリです。"),
    Document(page_content="LangChainはLLMアプリケーション開発を高速化します。"),
    Document(page_content="RAGは検索と生成を組み合わせた手法です。"),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def refine_query_node(state: RAGState) -> RAGState:
    """ユーザーの質問を検索に適した形式に改善する"""
    prompt = f"次の質問を、検索エンジンに適したキーワードに変換してください。元の質問: {state['question']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    state["refined_query"] = response.content
    return state

def retrieve_node(state: RAGState) -> RAGState:
    """改善されたクエリで検索を実行"""
    results = vectorstore.similarity_search(state["refined_query"], k=2)
    state["retrieved_docs"] = [doc.page_content for doc in results]
    return state

def generate_answer_node(state: RAGState) -> RAGState:
    """検索結果を基に回答を生成"""
    context = "\n".join(state["retrieved_docs"])
    prompt = f"以下の文脈を参考にして質問に答えてください。\n\n文脈:\n{context}\n\n質問: {state['question']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    state["answer"] = response.content
    return state

# グラフの構築
graph = StateGraph(RAGState)
graph.add_node("refine_query", refine_query_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_answer_node)

graph.add_edge(START, "refine_query")
graph.add_edge("refine_query", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# 実行
result = app.invoke({
    "question": "複雑なフローを作るツールは何？",
    "refined_query": "",
    "retrieved_docs": [],
    "answer": ""
})

print(f"元の質問: {result['question']}")
print(f"改善されたクエリ: {result['refined_query']}")
print(f"回答: {result['answer']}")
```

このパターンでは、曖昧な質問を検索に適した形式に変換してから検索を実行します。

## ベクトルデータベースの選定

注意点としてベクトルDB選定と検索精度と生成の信頼性を検討します。

### 主要なベクトルデータベース

プロダクション環境では、適切なベクトルデータベースの選定が重要です。

#### FAISS（Facebook AI Similarity Search）

- **特徴**: Meta が開発したオープンソースライブラリ
- **メリット**: 高速、ローカル実行可能、無料
- **デメリット**: スケーラビリティに限界、永続化は手動
- **適用場面**: プロトタイピング、小規模データ（数万～数十万ベクトル）

#### Pinecone

- **特徴**: マネージドベクトルデータベースサービス
- **メリット**: スケーラブル、メタデータフィルタリング、リアルタイム更新
- **デメリット**: 有料（無料枠あり）、外部サービス依存
- **適用場面**: 本番運用、大規模データ

#### Weaviate

- **特徴**: オープンソースのベクトルデータベース
- **メリット**: セルフホスト可能、GraphQL API、ハイブリッド検索
- **デメリット**: インフラ管理が必要
- **適用場面**: オンプレミス運用、カスタマイズ重視

#### Chroma

- **特徴**: 軽量なオープンソースベクトルデータベース
- **メリット**: 簡単なセットアップ、Python ネイティブ
- **デメリット**: 大規模データには不向き
- **適用場面**: 開発環境、中小規模アプリケーション

### 選定基準

ベクトルデータベースを選定する際の基準：

1. **データ規模**: 扱うベクトルの数（数千～数億）
2. **更新頻度**: データの追加・削除がどれくらい頻繁か
3. **レイテンシ要件**: 検索応答時間の要求（ミリ秒～秒）
4. **コスト**: 予算制約
5. **運用形態**: クラウド/オンプレミス/ハイブリッド

## 検索精度の向上

検索精度と生成の信頼性とスケーラビリティを考慮します。

### ハイブリッド検索

ベクトル検索とキーワード検索を組み合わせることで、精度を向上できます。

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# ベクトル検索リトリーバー
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# BM25（キーワード検索）リトリーバー
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# ハイブリッド検索（両者を組み合わせ）
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # 重み付け
)

# ハイブリッド検索を使用
results = ensemble_retriever.invoke("営業時間")
```

### メタデータフィルタリング

メタデータを活用して、検索範囲を絞り込みます。

```python
# メタデータ付きでベクトルストアを構築
documents_with_metadata = [
    Document(page_content="製品Aの仕様...", metadata={"category": "product", "year": 2024}),
    Document(page_content="2023年の売上...", metadata={"category": "financial", "year": 2023}),
]

vectorstore = FAISS.from_documents(documents_with_metadata, embeddings)

# メタデータでフィルタリング
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3, "filter": {"category": "product"}}
)
```

### リランキング

検索結果を再評価して、より関連性の高いものを上位に並べ替えます。

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# ベースリトリーバー
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# LLMを使ったリランキング
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# より関連性の高い結果が返される
results = compression_retriever.invoke("製品の保証について")
```

## 生成の信頼性向上

生成された回答の信頼性を高める手法を紹介します。

### ソース引用の追加

回答に引用元を含めることで、信頼性を向上させます。

```python
template = """以下の文脈を参考にして、質問に答えてください。
回答の最後に、参照した情報源を明記してください。

文脈:
{context}

質問: {question}

回答（情報源を含めて）:"""

prompt = ChatPromptTemplate.from_template(template)

# RAGチェーン構築（ソース引用付き）
rag_chain_with_sources = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain_with_sources.invoke("営業時間を教えてください")
print(answer)
# 出力例: 営業時間は平日9時から18時までです。（情報源: FAQ文書）
```

### 回答の検証

生成された回答が文脈と矛盾していないか検証します。

```python
def verify_answer(question: str, context: str, answer: str) -> dict:
    """回答が文脈に基づいているか検証"""
    verification_prompt = f"""
    文脈: {context}
    質問: {question}
    回答: {answer}

    この回答は文脈に基づいていますか？回答が文脈から導き出せる場合は「はい」、そうでない場合は「いいえ」と答えてください。
    """

    response = llm.invoke([HumanMessage(content=verification_prompt)])
    is_valid = "はい" in response.content

    return {
        "is_valid": is_valid,
        "explanation": response.content
    }

# 使用例
context = "営業時間は平日9時から18時までです。"
question = "営業時間を教えてください"
answer = "営業時間は24時間です。"

verification = verify_answer(question, context, answer)
print(f"回答の妥当性: {verification['is_valid']}")
print(f"説明: {verification['explanation']}")
```

## スケーラビリティの考慮

大規模データと高負荷に対応するための設計を検討します。

### バッチ処理による埋め込み生成

大量のドキュメントを効率的にベクトル化します。

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# バッチサイズを指定して効率化
documents_large = [Document(page_content=f"ドキュメント {i}") for i in range(1000)]

# バッチ処理（OpenAI APIのレート制限を考慮）
batch_size = 100
for i in range(0, len(documents_large), batch_size):
    batch = documents_large[i:i+batch_size]
    # ベクトルストアに追加
    vectorstore.add_documents(batch)
    print(f"処理済み: {i+len(batch)}/{len(documents_large)}")
```

### キャッシング戦略

頻繁に検索されるクエリをキャッシュして、レスポンスを高速化します。

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str) -> str:
    """検索結果をキャッシュ"""
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

# 同じクエリは高速に応答
result1 = cached_search("営業時間")  # 初回: ベクトル検索実行
result2 = cached_search("営業時間")  # 2回目: キャッシュから取得（高速）
```

### 非同期処理

複数の検索や LLM 呼び出しを並列化します。

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

async def async_rag_query(question: str) -> str:
    """非同期でRAGクエリを実行"""
    # 検索
    docs = await vectorstore.asimilarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # LLM呼び出し
    prompt = f"文脈: {context}\n質問: {question}\n回答:"
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content

# 複数のクエリを並列実行
async def main():
    questions = ["営業時間は？", "製品Aの仕様は？", "配送期間は？"]
    results = await asyncio.gather(*[async_rag_query(q) for q in questions])
    for q, r in zip(questions, results):
        print(f"Q: {q}\nA: {r}\n")

# 実行
# asyncio.run(main())
```

## 注意点とトレードオフ

RAG システムを構築する際の注意点を整理します。

### 検索精度とコストのトレードオフ

検索で取得するドキュメント数（k 値）を増やすと精度は向上しますが、LLM に渡すトークン数が増えてコストが上昇します。
適切な k 値を実験的に決定する必要があります。

### ベクトルDB運用コスト

マネージドサービス（Pinecone など）は便利ですが、データ量に応じてコストが増加します。
データ規模とクエリ頻度を見積もり、コストを予測しましょう。

### リアルタイム更新の必要性

ドキュメントが頻繁に更新される場合、ベクトルストアの再構築が必要です。
増分更新をサポートするベクトルDB を選定するか、更新頻度を下げる設計を検討します。

## 本章のまとめ

本章では、RAG の基本概念、ベクトル検索の仕組み、LangChain と LangGraph を使った RAG の実装パターンを解説しました。
RAG はエージェントに外部知識を供給する重要な技術であり、適切なベクトルデータベースの選定と検索精度の向上が鍵となります。
次章では、ビジネス視点からサービス企画と設計を考えます。

