# Camada 1 — Fundação de Conhecimento / Busca Semântica

**Data:** 2026-06-24
**Status:** Aprovado (design) — pendente revisão do spec pelo usuário

## Contexto e motivação

O objetivo geral é substituir uma solução baseada no Microsoft Copilot Studio,
diante do risco do contrato com a Microsoft terminar e o chatbot + plugins
ficarem inutilizáveis. O Copilot Studio agrega, na prática, três sistemas:

1. **Camada de conhecimento / busca** (este documento)
2. **Camada de RAG conversacional** (LLM gerando respostas fundamentadas)
3. **Camada de plugins / ações** (assistente chama APIs e dispara workflows)

A estratégia é construir nessa ordem, cada camada com seu próprio ciclo
spec → plano → implementação. Cada camada entrega valor por si só.

Princípio transversal: **evitar lock-in.** Assim como fugimos do lock-in da
Microsoft, não nos amarramos a um banco específico. Por isso a Camada 1 expõe
interfaces de abstração e mantém uma única implementação concreta por enquanto.

## Escopo desta camada

Um buscador semântico funcional e independente: ingere documentos de tipos
variados, gera embeddings localmente, armazena em Postgres + pgvector e oferece
busca híbrida (semântica + textual) via API REST.

### Fora de escopo (camadas futuras)
- Geração de respostas por LLM (Camada 2)
- Cache / atalho de respostas da LLM (Camada 2)
- Plugins / ações (Camada 3)

## Decisões de arquitetura

| Decisão | Escolha | Motivo |
|---|---|---|
| Banco | PostgreSQL + pgvector | Projeto em início, <1k docs; um único banco, sem cluster. pgvector escala bem além disso com HNSW. |
| Hospedagem | Self-hosted via Docker Compose | Controle total, sem lock-in de cloud. |
| Embeddings | Modelo local (Ollama + bge-m3) | Dados não saem do ambiente; multilíngue, bom em português; container pronto. |
| Stack da app | Python + FastAPI | Ecossistema rico para IA, embeddings e processamento de documentos. |
| Escala alvo | < 1.000 documentos | CPU suficiente, sem GPU. |
| Busca | Híbrida (vetorial + full-text) com RRF | Qualidade semântica + palavra-chave sem calibrar pesos. |

## Arquitetura geral (Docker Compose)

Três contêineres, todos no ambiente do usuário:

```
┌─────────────┐     HTTP      ┌──────────────┐
│   api       │──────────────▶│   ollama     │  embeddings (bge-m3, 1024 dim)
│  (FastAPI)  │               └──────────────┘
│             │     SQL       ┌──────────────┐
│             │──────────────▶│  postgres    │  pgvector + full-text
└─────────────┘               └──────────────┘
```

- **postgres** — imagem `pgvector/pgvector` (Postgres com a extensão pronta).
- **ollama** — serve o modelo `bge-m3`, expõe API HTTP de embeddings.
- **api** — aplicação FastAPI: ingestão, busca e as abstrações.

## Interfaces de abstração (desacoplamento)

O resto da aplicação só conhece estas interfaces. Hoje existe uma implementação
concreta de cada. Trocar de banco/embedding no futuro = nova classe, sem mexer
no resto. **Não** construímos adaptadores para tecnologias não usadas (YAGNI).

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class KnowledgeStore(Protocol):
    def index_document(self, doc: Document) -> str: ...
    def semantic_search(self, query: str, k: int) -> list[Hit]: ...
    def keyword_search(self, query: str, k: int) -> list[Hit]: ...
    def hybrid_search(self, query: str, k: int) -> list[Hit]: ...
    def delete_document(self, doc_id: str) -> None: ...
```

Implementações iniciais: `OllamaEmbedder`, `PostgresKnowledgeStore`.

## Schema do Postgres

```sql
CREATE EXTENSION IF NOT EXISTS vector;

documents(
  id            uuid primary key,
  source_type   text,          -- 'pdf' | 'office' | 'web' | 'structured' | 'faq'
  source_uri    text,
  title         text,
  metadata      jsonb,
  created_at    timestamptz default now()
)

chunks(
  id            uuid primary key,
  document_id   uuid references documents(id) on delete cascade,
  chunk_index   int,
  content       text,
  embedding     vector(1024),  -- bge-m3
  tsv           tsvector,      -- full-text em português
  unique(document_id, chunk_index)
)

-- índice HNSW para busca vetorial
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
-- índice GIN para busca textual
CREATE INDEX ON chunks USING gin (tsv);
```

`delete_document` remove o documento; os chunks caem por `ON DELETE CASCADE`.

## Pipeline de ingestão

```
arquivo / registro
   → loader (selecionado por tipo)
   → extrai texto
   → chunker (trechos com sobreposição)
   → embedder.embed(chunks)        [chama Ollama]
   → store.index_document(...)     [grava texto + vetor + tsv]
```

### Loaders (módulos isolados e testáveis, libs open prontas)
- **PDF:** `pypdf` (ou `unstructured` para layouts complexos)
- **Office:** `python-docx` (.docx), `python-pptx` (.pptx)
- **Web / wiki:** `trafilatura` (ou `BeautifulSoup`)
- **Estruturado:** parser direto de JSON/CSV
- **FAQ:** par pergunta→resposta como registro estruturado curado

### Chunking
Divisão por tamanho com sobreposição (overlap), preservando limites de
sentença quando possível. Parâmetros (tamanho do chunk, overlap) configuráveis.

### FAQ
Ingerida como `source_type = 'faq'` (conteúdo curado). Recebe **boost de
ranking** na busca, por ser resposta validada. O atalho que devolve a FAQ
direto sem chamar a LLM é item da Camada 2 (lá existe a chamada cara a evitar).

## Busca híbrida

A consulta dispara as duas buscas e funde os resultados com **RRF (Reciprocal
Rank Fusion)**:

```
query
  → embedder.embed([query])  → vetor
  → semantic_search (pgvector, cosseno)  →  ranking A
  → keyword_search  (tsvector, ts_rank)  →  ranking B
  → fusão RRF (com boost para FAQ)       →  resultado final
```

RRF combina rankings por posição (1/(k+rank)), dispensando calibração de pesos
entre as duas escalas de score. Retorna trechos + score + origem (documento).

## API REST

```
POST   /documents        ingere um documento (upload ou referência)
GET    /search?q=&k=     busca híbrida → trechos + score + origem
DELETE /documents/{id}   remove documento e seus chunks
GET    /health           status (conexão db + disponibilidade ollama)
```

## Tratamento de erros

- **Ingestão resiliente por documento:** falha em um arquivo (ex.: PDF
  corrompido) não derruba o lote; registra o erro e segue.
- **Chamadas ao Ollama:** retry com backoff; `/health` reporta indisponível.
- **Validação de entrada** na API (tipos de arquivo, tamanho, parâmetros).

## Estratégia de testes (TDD)

- **Unitários:** chunker (limites, overlap), `OllamaEmbedder` (Ollama mockado),
  fusão RRF, seleção de loader por tipo.
- **Integração:** ciclo completo *ingerir → buscar* contra um Postgres de teste
  real (ex.: testcontainers ou Postgres efêmero no Docker).
- **Resiliência:** ingestão de lote com um documento inválido no meio.

## Estrutura de módulos (proposta)

```
app/
  interfaces.py          # Embedder, KnowledgeStore, Document, Hit
  embedders/ollama.py    # OllamaEmbedder
  store/postgres.py      # PostgresKnowledgeStore (+ RRF)
  ingest/
    loaders/             # pdf, office, web, structured, faq
    chunker.py
    pipeline.py
  api/main.py            # rotas FastAPI
  config.py
docker-compose.yml
tests/
```

## Critérios de sucesso

1. `docker compose up` sobe os três serviços.
2. Ingerir um PDF, uma página web, um registro estruturado e uma FAQ funciona.
3. Uma busca semântica retorna trechos relevantes por significado (não só
   palavra exata); a busca textual encontra termos exatos; a híbrida combina.
4. FAQs curadas aparecem priorizadas quando relevantes.
5. Toda a aplicação fala apenas com `KnowledgeStore` e `Embedder` — nenhum
   acesso direto a Postgres ou Ollama fora dos adaptadores.
6. Nenhum dado de documento sai do ambiente (embeddings gerados localmente).
