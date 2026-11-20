# LangChain and LlamaIndex adapters

ragpick works alongside LangChain and LlamaIndex rather than replacing them. Any of their
loaders, embedding models, retrievers and LLMs can be used as a ragpick component, with
ragpick's config, access control, selector and evaluation around it.

Neither framework is a ragpick dependency. The adapters only call public methods
(`embed_documents`, `invoke`, `get_text_embedding_batch`, `retrieve`, …), so install the
LangChain or LlamaIndex packages you want yourself:

```bash
pip install ragpick langchain-openai          # or: pip install ragpick llama-index
```

| Kind | `type: langchain` wraps | `type: llamaindex` wraps |
|---|---|---|
| `loader` | a document loader (`lazy_load()` / `load()`) | a reader (`load_data()`), default `SimpleDirectoryReader` |
| `embedder` | `Embeddings` (`embed_documents`, `embed_query`) | an embedding model (`get_text_embedding_batch`, `get_query_embedding`) |
| `retriever` | a retriever (`invoke(query)`) | a retriever (`retrieve(query)`) |
| `generator` | a chat model (`invoke(messages)`) | an LLM (`complete(prompt)`) |

## From YAML

Give the class's import path and its constructor arguments:

```yaml
blueprint: hybrid
components:
  loader:
    type: langchain
    params:
      class_path: langchain_community.document_loaders.PyPDFLoader
      path_arg: file_path               # the constructor argument that receives the path
  embedder:
    type: langchain
    params:
      class_path: langchain_openai.OpenAIEmbeddings
      kwargs: {model: text-embedding-3-small}
  generator:
    type: llamaindex
    params:
      class_path: llama_index.llms.openai.OpenAI
      kwargs: {model: gpt-4o-mini, temperature: 0}
```

`class_path` accepts `package.module.Class` or `package.module:Class`. A missing package
fails with a clear "could not import … install the package" message.

!!! warning "Treat configs as code"
    `class_path` imports and instantiates arbitrary Python classes. Only load `rag.yaml`
    files you trust, just as you would only run code you trust.

## From Python

Pass objects you've already built:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragpick import Pipeline, load_config
from ragpick.adapters import LangChainEmbedder, LangChainGenerator

pipeline = Pipeline(
    load_config("rag.yaml"),
    components={
        "embedder": LangChainEmbedder(embeddings=OpenAIEmbeddings(model="text-embedding-3-small")),
        "generator": LangChainGenerator(chat_model=ChatOpenAI(model="gpt-4o-mini")),
    },
)
```

## Details

- **Generators** use ragpick's grounded prompt and citation parsing, and also work as an
  evaluation judge (`eval.judge: {type: langchain, params: {...}}`). LangChain token usage
  comes from `usage_metadata`; LlamaIndex usage is estimated.
- **Embedders** use the framework's query-embedding method for searches and its
  document-embedding method for indexing.
- **Loaders** turn `source` / `file_path` metadata into document ids relative to the
  ingested path. When one file yields several documents (PDF pages), `:<n>` is appended.
- **Retrievers** are for indexes managed *outside* ragpick. External retrievers can't apply
  ragpick filters, so filters and `allowed_groups` access control are applied to their
  results afterwards (post-filtering). Configure the external retriever to return enough
  candidates for that. Missing scores become `1 / rank`.
