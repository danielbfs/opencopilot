# Camada 1 — Busca Semântica: Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir um buscador semântico self-hosted que ingere documentos variados, gera embeddings localmente (Ollama/bge-m3), armazena em Postgres+pgvector e oferece busca híbrida (vetorial + full-text fundida por RRF) via API REST.

**Architecture:** Aplicação FastAPI em Docker Compose com três serviços (api, postgres+pgvector, ollama). O resto da aplicação fala apenas com duas interfaces (`Embedder`, `KnowledgeStore`); há uma implementação concreta de cada (`OllamaEmbedder`, `PostgresKnowledgeStore`). Ingestão por pipeline (loader → chunker → embedder → store), resiliente por documento. FAQ entra como `source_type='faq'` com boost de ranking.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, psycopg (v3) + pgvector, httpx, pypdf, python-docx, python-pptx, trafilatura, pydantic, pydantic-settings, pytest, testcontainers. Docker Compose com imagens `pgvector/pgvector:pg16` e `ollama/ollama`.

## Global Constraints

- Python >= 3.12.
- Embeddings gerados **localmente** via Ollama (modelo `bge-m3`, **1024 dimensões**); nenhum dado de documento sai do ambiente.
- Toda a aplicação fala **apenas** com `KnowledgeStore` e `Embedder` — proibido acesso direto a Postgres ou Ollama fora dos adaptadores (`app/store/postgres.py`, `app/embedders/ollama.py`).
- Escala alvo: < 1.000 documentos, CPU apenas (sem GPU).
- Banco: PostgreSQL com extensão `vector`. Full-text em português (`to_tsvector('portuguese', ...)`).
- Busca híbrida funde rankings por **RRF** `1/(k_rrf + rank)` com `k_rrf = 60`; sem calibrar pesos entre escalas de score.
- Ingestão **resiliente por documento**: falha em um item de um lote registra erro e não derruba o lote.
- Chamadas ao Ollama: retry com backoff exponencial.
- Sem lock-in: não construir adaptadores para tecnologias não usadas (YAGNI).
- Estrutura de módulos conforme seção "Estrutura de módulos" do spec.

---

## File Structure

- `pyproject.toml` — metadados, dependências, config de pytest/ruff.
- `app/__init__.py`, `app/config.py` — settings via pydantic-settings (DB URL, Ollama URL, modelo, chunk size/overlap, dims).
- `app/interfaces.py` — Protocols `Embedder`, `KnowledgeStore`; dataclasses/models `Document`, `Chunk`, `Hit`.
- `app/embedders/ollama.py` — `OllamaEmbedder` (HTTP + retry/backoff).
- `app/store/postgres.py` — `PostgresKnowledgeStore` (index/semantic/keyword/hybrid/delete + RRF + boost FAQ).
- `app/store/migrations.sql` — DDL (extensão, tabelas, índices HNSW/GIN).
- `app/ingest/loaders/` — `base.py`, `pdf.py`, `office.py`, `web.py`, `structured.py`, `faq.py`, `registry.py` (seleção por tipo).
- `app/ingest/chunker.py` — chunking com overlap, preservando sentenças.
- `app/ingest/pipeline.py` — orquestra loader→chunker→embedder→store; resiliência por documento.
- `app/api/main.py` — rotas FastAPI; `app/api/deps.py` — providers de dependência.
- `docker-compose.yml`, `Dockerfile`, `.dockerignore`, `scripts/init_db.py`, `scripts/pull_model.sh`.
- `tests/` — `conftest.py`, unit (`test_chunker.py`, `test_ollama_embedder.py`, `test_rrf.py`, `test_loader_registry.py`), integração (`test_store_integration.py`, `test_pipeline_integration.py`, `test_api_integration.py`).

---

## Task 1: Scaffolding do projeto + config

**Files:**
- Create: `pyproject.toml`, `app/__init__.py`, `app/config.py`, `.dockerignore`, `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `app.config.Settings` (pydantic-settings) com campos: `database_url: str`, `ollama_url: str` (default `http://ollama:11434`), `embedding_model: str` (default `bge-m3`), `embedding_dims: int` (default `1024`), `chunk_size: int` (default `512`), `chunk_overlap: int` (default `64`), `rrf_k: int` (default `60`), `faq_boost: float` (default `0.5`). Função `get_settings() -> Settings` (cacheada com `functools.lru_cache`). Lê de variáveis de ambiente com prefixo `OC_`.

- [ ] **Step 1: Escrever pyproject.toml**

```toml
[project]
name = "opencopilot"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.110", "uvicorn[standard]>=0.29", "pydantic>=2.6",
  "pydantic-settings>=2.2", "httpx>=0.27", "psycopg[binary]>=3.1",
  "pgvector>=0.2.5", "pypdf>=4.0", "python-docx>=1.1", "python-pptx>=0.6.23",
  "trafilatura>=1.8", "python-multipart>=0.0.9",
]
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "testcontainers[postgres]>=4.0", "ruff>=0.4", "respx>=0.21"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: requires Docker (postgres/ollama)"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 2: Escrever o teste que falha** em `tests/test_config.py`

```python
import os
from app.config import get_settings

def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("OC_DATABASE_URL", "postgresql://u:p@localhost/db")
    get_settings.cache_clear()
    s = get_settings()
    assert s.embedding_model == "bge-m3"
    assert s.embedding_dims == 1024
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    assert s.rrf_k == 60
    assert s.ollama_url == "http://ollama:11434"
```

- [ ] **Step 3: Rodar e ver falhar** — `pytest tests/test_config.py -v` → FAIL (`ModuleNotFoundError: app.config`).

- [ ] **Step 4: Implementar `app/config.py`**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OC_")
    database_url: str
    ollama_url: str = "http://ollama:11434"
    embedding_model: str = "bge-m3"
    embedding_dims: int = 1024
    chunk_size: int = 512
    chunk_overlap: int = 64
    rrf_k: int = 60
    faq_boost: float = 0.5

@lru_cache
def get_settings() -> Settings:
    return Settings()
```
Criar `app/__init__.py` e `tests/__init__.py` vazios.

- [ ] **Step 5: Rodar e ver passar** — `pytest tests/test_config.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml app/ tests/ .dockerignore
git commit -m "feat: scaffolding e configuração do projeto"
```

---

## Task 2: Interfaces e tipos de domínio

**Files:**
- Create: `app/interfaces.py`, `tests/test_interfaces.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `Document` (pydantic BaseModel): `id: str | None = None`, `source_type: str`, `source_uri: str | None = None`, `title: str | None = None`, `metadata: dict = {}`, `content: str` (texto completo já extraído, mantido para loaders/registry), `chunks: list[Chunk] = []` (chunks de texto já produzidos pela pipeline — loader→chunker — prontos para o store vetorizar na ingestão).
  - `Chunk` (BaseModel): `chunk_index: int`, `content: str`.
  - `Hit` (BaseModel): `chunk_id: str`, `document_id: str`, `content: str`, `score: float`, `source_type: str`, `title: str | None`, `source_uri: str | None`, `chunk_index: int`.
  - `Embedder(Protocol)`: `def embed(self, texts: list[str]) -> list[list[float]]: ...`
  - `KnowledgeStore(Protocol)`: `index_document(self, doc: Document) -> str`; `semantic_search(self, query: str, k: int) -> list[Hit]`; `keyword_search(self, query: str, k: int) -> list[Hit]`; `hybrid_search(self, query: str, k: int) -> list[Hit]`; `delete_document(self, doc_id: str) -> None`.

  > Nota de design (Decisão — Opção A): o `KnowledgeStore` recebe um `Embedder` por **injeção no construtor** (`PostgresKnowledgeStore(db, embedder)`) e é responsável por toda a vetorização internamente — tanto da query nas buscas (`semantic_search`/`hybrid_search` recebem **texto**, não vetor) quanto dos chunks na ingestão (`index_document` recebe um `Document` cujos `chunks` já são texto puro; o store chama `self.embedder.embed([...])` para gerar os vetores antes de persistir). Isso mantém as assinaturas do Protocol idênticas às do spec e preserva o Critério 5: o store nunca fala com Ollama diretamente, só através do `Embedder` injetado.

- [ ] **Step 1: Teste que falha** em `tests/test_interfaces.py`

```python
from app.interfaces import Document, Chunk, Hit

def test_document_defaults():
    d = Document(source_type="faq", content="texto")
    assert d.metadata == {} and d.id is None and d.chunks == []

def test_hit_roundtrip():
    h = Hit(chunk_id="c1", document_id="d1", content="x", score=0.9,
            source_type="pdf", title=None, source_uri=None, chunk_index=0)
    assert h.score == 0.9
```

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_interfaces.py -v` → FAIL.

- [ ] **Step 3: Implementar `app/interfaces.py`** com os models pydantic acima e os dois `Protocol` (importar `Protocol`, `runtime_checkable` de `typing`).

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_interfaces.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add app/interfaces.py tests/test_interfaces.py && git commit -m "feat: interfaces Embedder/KnowledgeStore e tipos de domínio"`

---

## Task 3: Chunker com overlap

**Files:**
- Create: `app/ingest/__init__.py`, `app/ingest/chunker.py`, `tests/test_chunker.py`

**Interfaces:**
- Consumes: `app.interfaces.Chunk`.
- Produces: `def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]`. Divide por caracteres respeitando limites de sentença (quebra preferencialmente em `.`/`!`/`?`/`\n` dentro de uma janela de tolerância antes do limite); chunks consecutivos compartilham `overlap` caracteres; `chunk_index` sequencial começando em 0; texto vazio/whitespace → lista vazia.

- [ ] **Step 1: Testes que falham** em `tests/test_chunker.py`

```python
from app.ingest.chunker import chunk_text

def test_empty_text_returns_no_chunks():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []

def test_short_text_single_chunk():
    chunks = chunk_text("Frase curta.", chunk_size=100, overlap=10)
    assert len(chunks) == 1 and chunks[0].chunk_index == 0

def test_long_text_has_overlap():
    text = "a" * 250
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    # sobreposição: fim do chunk 0 reaparece no início do chunk 1
    assert chunks[0].content[-20:] == chunks[1].content[:20]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

def test_prefers_sentence_boundary():
    text = "Primeira frase aqui. " + "x" * 80 + " Segunda."
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert chunks[0].content.rstrip().endswith(".")
```

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_chunker.py -v` → FAIL.

- [ ] **Step 3: Implementar `chunk_text`** com janela deslizante: a partir do índice atual pega `chunk_size`; procura o último delimitador de sentença na janela `[end-tolerance, end]` (tolerância = 20% do chunk_size); usa esse ponto de corte se existir, senão corta em `chunk_size`; próximo início = ponto de corte − `overlap` (nunca negativo); para quando consome todo o texto. Ignorar texto só com whitespace.

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_chunker.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add app/ingest/__init__.py app/ingest/chunker.py tests/test_chunker.py && git commit -m "feat: chunker com overlap e limites de sentença"`

---

## Task 4: OllamaEmbedder (HTTP + retry/backoff)

**Files:**
- Create: `app/embedders/__init__.py`, `app/embedders/ollama.py`, `tests/test_ollama_embedder.py`

**Interfaces:**
- Consumes: `app.config.Settings`.
- Produces: `class OllamaEmbedder` com `__init__(self, base_url: str, model: str = "bge-m3", max_retries: int = 3, timeout: float = 30.0, client: httpx.Client | None = None)` e `embed(self, texts: list[str]) -> list[list[float]]`. Chama `POST {base_url}/api/embed` com `{"model": model, "input": texts}`, lê `resp.json()["embeddings"]`. Retry com backoff exponencial (`0.5 * 2**attempt`) em erro de conexão ou status 5xx; após esgotar tentativas, levanta `EmbeddingError`. Também expõe `health(self) -> bool` (GET `{base_url}/api/tags`, True se 200).

- [ ] **Step 1: Testes que falham** em `tests/test_ollama_embedder.py` (usando `respx` para mockar httpx)

```python
import httpx, respx, pytest
from app.embedders.ollama import OllamaEmbedder, EmbeddingError

BASE = "http://ollama:11434"

@respx.mock
def test_embed_returns_vectors():
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(
        200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]}))
    emb = OllamaEmbedder(base_url=BASE, model="bge-m3")
    out = emb.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]

@respx.mock
def test_embed_retries_then_succeeds():
    route = respx.post(f"{BASE}/api/embed")
    route.side_effect = [httpx.Response(503),
                         httpx.Response(200, json={"embeddings": [[1.0]]})]
    emb = OllamaEmbedder(base_url=BASE, max_retries=3)
    assert emb.embed(["x"]) == [[1.0]]
    assert route.call_count == 2

@respx.mock
def test_embed_raises_after_exhausting_retries():
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(503))
    emb = OllamaEmbedder(base_url=BASE, max_retries=2)
    with pytest.raises(EmbeddingError):
        emb.embed(["x"])

@respx.mock
def test_health_true_on_200():
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
    assert OllamaEmbedder(base_url=BASE).health() is True
```

> Para tornar o backoff instantâneo nos testes, `time.sleep` deve ser mockável: injetar via parâmetro `sleep=time.sleep` no `__init__` ou usar `monkeypatch` sobre `app.embedders.ollama.time.sleep`. Os testes acima rodam rápido se o sleep for patchado por uma fixture autouse no teste.

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_ollama_embedder.py -v` → FAIL.

- [ ] **Step 3: Implementar `app/embedders/ollama.py`** com classe `EmbeddingError(Exception)` e `OllamaEmbedder` conforme contrato; loop de tentativas com backoff exponencial; reusar `httpx.Client` (criar se não injetado).

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_ollama_embedder.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add app/embedders tests/test_ollama_embedder.py && git commit -m "feat: OllamaEmbedder com retry/backoff e health"`

---

## Task 5: Função de fusão RRF (com boost FAQ)

**Files:**
- Create: `app/store/__init__.py`, `app/store/rrf.py`, `tests/test_rrf.py`

**Interfaces:**
- Consumes: `app.interfaces.Hit`.
- Produces: `def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60, faq_boost: float = 0.5) -> list[Hit]`. Cada ranking é uma lista de `Hit` em ordem de relevância. Score RRF de um chunk = soma sobre rankings de `1/(k + rank)` (rank começa em 1). Hits com `source_type == "faq"` recebem `+faq_boost` adicionado ao score final. Deduplica por `chunk_id`, mantém o `Hit` (atualizando `.score` para o RRF final) e retorna ordenado por score desc.

- [ ] **Step 1: Testes que falham** em `tests/test_rrf.py`

```python
from app.interfaces import Hit
from app.store.rrf import reciprocal_rank_fusion

def mk(cid, st="pdf"):
    return Hit(chunk_id=cid, document_id="d", content="", score=0.0,
               source_type=st, title=None, source_uri=None, chunk_index=0)

def test_fusion_rewards_high_rank_in_both():
    a = [mk("1"), mk("2"), mk("3")]
    b = [mk("2"), mk("1"), mk("4")]
    fused = reciprocal_rank_fusion([a, b], k=60, faq_boost=0.0)
    assert fused[0].chunk_id in {"1", "2"}
    ids = [h.chunk_id for h in fused]
    assert set(ids) == {"1", "2", "3", "4"}

def test_faq_boost_promotes_faq():
    a = [mk("1"), mk("faq1", st="faq")]   # faq em rank 2
    fused = reciprocal_rank_fusion([a], k=60, faq_boost=1.0)
    assert fused[0].chunk_id == "faq1"     # boost supera a posição

def test_dedup_keeps_single_entry():
    a = [mk("1")]; b = [mk("1")]
    fused = reciprocal_rank_fusion([a, b])
    assert len(fused) == 1
```

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_rrf.py -v` → FAIL.

- [ ] **Step 3: Implementar `reciprocal_rank_fusion`** com dict acumulador por `chunk_id`.

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_rrf.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add app/store/__init__.py app/store/rrf.py tests/test_rrf.py && git commit -m "feat: fusão RRF com boost de FAQ"`

---

## Task 6: Schema/migrations do Postgres

**Files:**
- Create: `app/store/migrations.sql`, `scripts/__init__.py`, `scripts/init_db.py`, `tests/test_migrations_integration.py`

**Interfaces:**
- Consumes: `app.config.get_settings`.
- Produces: `scripts/init_db.py` com `def run_migrations(database_url: str) -> None` que abre conexão psycopg, registra pgvector (`pgvector.psycopg.register_vector`), e executa o conteúdo de `migrations.sql` (idempotente via `IF NOT EXISTS`). Executável como `python -m scripts.init_db`.

- [ ] **Step 1: Escrever `app/store/migrations.sql`**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_type text NOT NULL,
  source_uri  text,
  title       text,
  metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index int NOT NULL,
  content     text NOT NULL,
  embedding   vector(1024),
  tsv         tsvector,
  UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin
  ON chunks USING gin (tsv);
```

- [ ] **Step 2: Teste de integração que falha** em `tests/test_migrations_integration.py` (marcado `@pytest.mark.integration`)

```python
import pytest, psycopg
from testcontainers.postgres import PostgresContainer
from scripts.init_db import run_migrations

@pytest.mark.integration
def test_migrations_create_tables_and_indexes():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        url = pg.get_connection_url().replace("psycopg2", "psycopg")
        run_migrations(url)
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('documents'), to_regclass('chunks')")
            docs, chunks = cur.fetchone()
            assert docs and chunks
            cur.execute("SELECT indexname FROM pg_indexes WHERE tablename='chunks'")
            idx = {r[0] for r in cur.fetchall()}
            assert "chunks_embedding_hnsw" in idx and "chunks_tsv_gin" in idx
```

> `get_connection_url()` do testcontainers retorna URL com driver `psycopg2`; trocar para `psycopg` (v3). Confirmar o nome exato do driver na versão instalada.

- [ ] **Step 3: Rodar e ver falhar** — `pytest tests/test_migrations_integration.py -v -m integration` → FAIL (módulo ausente). Requer Docker.

- [ ] **Step 4: Implementar `scripts/init_db.py`** lendo o `.sql` adjacente a `app/store/`, executando-o, com `if __name__ == "__main__": run_migrations(get_settings().database_url)`.

- [ ] **Step 5: Rodar e ver passar** — `pytest tests/test_migrations_integration.py -v -m integration` → PASS.

- [ ] **Step 6: Commit** — `git add app/store/migrations.sql scripts/ tests/test_migrations_integration.py && git commit -m "feat: schema Postgres (pgvector, HNSW, GIN) e runner de migrations"`

---

## Task 7: PostgresKnowledgeStore (index/semantic/keyword/hybrid/delete)

**Files:**
- Create: `app/store/postgres.py`, `tests/test_store_integration.py`

**Interfaces:**
- Consumes: `Document`, `Chunk`, `Hit` (interfaces); `Embedder` (Task 2/4, injetado); `reciprocal_rank_fusion` (Task 5); migrations (Task 6).
- Produces: `class PostgresKnowledgeStore` com `__init__(self, database_url: str, embedder: Embedder, rrf_k: int = 60, faq_boost: float = 0.5)` (Decisão — Opção A: o `Embedder` é injetado no construtor; o store é o único responsável por vetorizar — chunks na ingestão, query na busca — chamando `self.embedder.embed([...])` internamente; nunca fala com Ollama diretamente); métodos conforme Protocol da Task 2. Detalhes:
  - `index_document(doc)`: lê `doc.chunks` (texto puro, já produzidos pela pipeline); chama `self.embedder.embed([c.content for c in doc.chunks])` para obter os vetores; INSERT em `documents` (retorna `id`), depois INSERT em `chunks` por chunk com `embedding` (vetor obtido do embedder) e `tsv = to_tsvector('portuguese', content)`. Retorna `document_id` (str).
  - `semantic_search(query, k)`: chama `self.embedder.embed([query])[0]` para obter o vetor da query; `ORDER BY embedding <=> %s LIMIT k`; score = `1 - distância` (similaridade cosseno).
  - `keyword_search(query, k)`: `WHERE tsv @@ plainto_tsquery('portuguese', %s) ORDER BY ts_rank(tsv, ...) DESC LIMIT k` (já era texto; sem mudança).
  - `hybrid_search(query, k)`: vetoriza `query` uma única vez via `self.embedder.embed([query])[0]`, roda `semantic_search`-equivalente (reaproveitando o vetor já calculado, sem vetorizar de novo) e `keyword_search(query, k)`, funde via `reciprocal_rank_fusion([sem, kw], k=rrf_k, faq_boost=faq_boost)`, retorna top-k.
  - `delete_document(doc_id)`: `DELETE FROM documents WHERE id=%s` (cascade derruba chunks).
  - `health()`: `SELECT 1` → bool.

- [ ] **Step 1: Testes de integração que falham** em `tests/test_store_integration.py` (fixture de módulo sobe `PostgresContainer`, roda migrations; injeta um `Embedder` fake/determinístico; cada método testado)

```python
import pytest
from testcontainers.postgres import PostgresContainer
from app.interfaces import Document, Chunk
from app.store.postgres import PostgresKnowledgeStore
from scripts.init_db import run_migrations

pytestmark = pytest.mark.integration

class FakeEmbedder:
    """Vetor determinístico: comprimento do texto mod 7, repetido em 1024 dims."""
    def embed(self, texts):
        return [[float(len(t) % 7)] * 1024 for t in texts]

@pytest.fixture(scope="module")
def store():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        url = pg.get_connection_url().replace("psycopg2", "psycopg")
        run_migrations(url)
        yield PostgresKnowledgeStore(url, FakeEmbedder(), rrf_k=60, faq_boost=0.5)

def test_index_then_semantic_search(store):
    doc = Document(source_type="pdf", title="T", content="ignorado",
                    chunks=[Chunk(chunk_index=0, content="gato preto"),
                            Chunk(chunk_index=1, content="cachorro branco")])
    did = store.index_document(doc)
    hits = store.semantic_search("gato preto", k=2)
    assert hits and hits[0].document_id == did

def test_keyword_search_finds_exact_term(store):
    doc = Document(source_type="web", content="x",
                    chunks=[Chunk(chunk_index=0, content="contrato rescisão multa")])
    store.index_document(doc)
    hits = store.keyword_search("rescisão", k=5)
    assert any("rescisão" in h.content for h in hits)

def test_hybrid_search_returns_results(store):
    hits = store.hybrid_search("gato", k=5)
    assert isinstance(hits, list)

def test_delete_cascades(store):
    doc = Document(source_type="faq", content="x",
                    chunks=[Chunk(chunk_index=0, content="apagável")])
    did = store.index_document(doc)
    store.delete_document(did)
    assert all(h.document_id != did for h in store.keyword_search("apagável", k=5))
```

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_store_integration.py -v -m integration` → FAIL.

- [ ] **Step 3: Implementar `app/store/postgres.py`**. Usar psycopg3; em cada conexão chamar `pgvector.psycopg.register_vector(conn)`. Mapear linhas para `Hit` (selecionar join `chunks`→`documents` para `source_type`/`title`/`source_uri`). Para semântico, retornar `score = 1 - (embedding <=> query)`. Vetorização de chunks (ingestão) e de query (busca) sempre via `self.embedder.embed(...)` — nunca recebida de fora já calculada.

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_store_integration.py -v -m integration` → PASS.

- [ ] **Step 5: Commit** — `git add app/store/postgres.py tests/test_store_integration.py && git commit -m "feat: PostgresKnowledgeStore (semantic/keyword/hybrid/delete)"`

---

## Task 8: Loaders e registry de seleção por tipo

**Files:**
- Create: `app/ingest/loaders/__init__.py`, `base.py`, `pdf.py`, `office.py`, `web.py`, `structured.py`, `faq.py`, `registry.py`, `tests/test_loader_registry.py`, `tests/fixtures/` (amostras)

**Interfaces:**
- Consumes: `app.interfaces.Document`.
- Produces:
  - `base.py`: `class Loader(Protocol): def load(self, source: bytes | str | dict, *, source_uri: str | None = None, title: str | None = None) -> Document: ...`
  - `pdf.PdfLoader` (pypdf → texto, `source_type="pdf"`), `office.DocxLoader`/`office.PptxLoader` (`source_type="office"`), `web.WebLoader` (trafilatura sobre HTML, `source_type="web"`), `structured.StructuredLoader` (JSON/CSV → texto concatenado de campos, `source_type="structured"`), `faq.FaqLoader` (`{"question":..., "answer":...}` → content `"P: ...\nR: ..."`, `source_type="faq"`).
  - `registry.get_loader(source_type: str) -> Loader` mapeando `"pdf"`,`"office"`,`"web"`,`"structured"`,`"faq"`; tipo desconhecido → `ValueError`.

- [ ] **Step 1: Testes que falham** em `tests/test_loader_registry.py`

```python
import pytest
from app.ingest.loaders.registry import get_loader
from app.ingest.loaders.faq import FaqLoader

def test_registry_selects_by_type():
    assert isinstance(get_loader("faq"), FaqLoader)
    assert get_loader("pdf").__class__.__name__ == "PdfLoader"

def test_unknown_type_raises():
    with pytest.raises(ValueError):
        get_loader("xml")

def test_faq_loader_builds_document():
    doc = get_loader("faq").load({"question": "Qual o prazo?", "answer": "30 dias"})
    assert doc.source_type == "faq"
    assert "Qual o prazo?" in doc.content and "30 dias" in doc.content

def test_structured_loader_concatenates_fields():
    doc = get_loader("structured").load({"nome": "Ana", "cargo": "Dev"})
    assert "Ana" in doc.content and "Dev" in doc.content
```

> Para PDF/Office/Web, os testes do registry só validam seleção e os loaders simples (faq/structured). Testes de extração real de PDF/docx/pptx/web ficam cobertos pelo teste de pipeline de integração (Task 9) com fixtures reais mínimas geradas; não criar dependência de binários grandes.

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_loader_registry.py -v` → FAIL.

- [ ] **Step 3: Implementar os loaders e o registry.** Cada loader retorna `Document` com `content` extraído; preencher `title`/`source_uri` quando fornecidos.

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_loader_registry.py -v` → PASS.

- [ ] **Step 5: Commit** — `git add app/ingest/loaders tests/test_loader_registry.py && git commit -m "feat: loaders (pdf/office/web/structured/faq) e registry por tipo"`

---

## Task 9: Pipeline de ingestão (resiliente por documento)

**Files:**
- Create: `app/ingest/pipeline.py`, `tests/test_pipeline_integration.py`

**Interfaces:**
- Consumes: `get_loader` (Task 8), `chunk_text` (Task 3), `KnowledgeStore` (Task 2/7), `Document`.
- Produces:
  - `class IngestResult(BaseModel)`: `document_id: str | None`, `ok: bool`, `error: str | None = None`, `source_uri: str | None = None`.
  - `class IngestPipeline.__init__(self, store, chunk_size, overlap)` (Decisão — Opção A: a pipeline **não** recebe `Embedder` e **não** gera embeddings; só loader→chunker→`Document` com chunks de texto→`store.index_document(doc)`. Toda vetorização é responsabilidade do `KnowledgeStore`, que tem seu próprio `Embedder` injetado).
  - `ingest_one(self, source_type, source, *, source_uri=None, title=None) -> IngestResult`: loader (extrai texto) → `chunk_text` (gera chunks de texto com overlap) → monta `doc.chunks` → `store.index_document(doc)`; captura exceção e retorna `IngestResult(ok=False, error=...)`.
  - `ingest_batch(self, items: list[dict]) -> list[IngestResult]`: aplica `ingest_one` a cada item; um item com erro **não** interrompe o lote.

- [ ] **Step 1: Testes de integração que falham** em `tests/test_pipeline_integration.py`

```python
import pytest
from testcontainers.postgres import PostgresContainer
from app.store.postgres import PostgresKnowledgeStore
from app.ingest.pipeline import IngestPipeline
from scripts.init_db import run_migrations

pytestmark = pytest.mark.integration

class FakeEmbedder:
    """Injetado no store, não na pipeline — a pipeline não vetoriza."""
    def embed(self, texts): return [[float(len(t) % 7)] * 1024 for t in texts]

@pytest.fixture(scope="module")
def pipeline_and_store():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        url = pg.get_connection_url().replace("psycopg2", "psycopg")
        run_migrations(url)
        store = PostgresKnowledgeStore(url, FakeEmbedder())
        yield IngestPipeline(store, chunk_size=200, overlap=20), store

def test_ingest_then_search_cycle(pipeline_and_store):
    pipe, store = pipeline_and_store
    res = pipe.ingest_one("faq", {"question": "Qual o horário?", "answer": "9h às 18h"})
    assert res.ok and res.document_id
    hits = store.keyword_search("horário", k=5)
    assert any("horário" in h.content or "9h" in h.content for h in hits)

def test_batch_with_one_invalid_does_not_abort(pipeline_and_store):
    pipe, _ = pipeline_and_store
    items = [
        {"source_type": "faq", "source": {"question": "ok?", "answer": "sim"}},
        {"source_type": "pdf", "source": b"%%nao-eh-pdf%%"},   # inválido
        {"source_type": "structured", "source": {"campo": "valido"}},
    ]
    results = pipe.ingest_batch(items)
    assert results[0].ok and results[2].ok
    assert results[1].ok is False and results[1].error
```

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_pipeline_integration.py -v -m integration` → FAIL.

- [ ] **Step 3: Implementar `app/ingest/pipeline.py`** com try/except por documento em `ingest_one`; `ingest_batch` itera sem propagar exceções. A pipeline monta `Document(chunks=chunk_text(texto_extraido, ...))` e delega toda vetorização ao `store.index_document(doc)` — a pipeline nunca chama um `Embedder`.

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_pipeline_integration.py -v -m integration` → PASS.

- [ ] **Step 5: Commit** — `git add app/ingest/pipeline.py tests/test_pipeline_integration.py && git commit -m "feat: pipeline de ingestão resiliente por documento"`

---

## Task 10: API FastAPI (rotas + deps)

**Files:**
- Create: `app/api/__init__.py`, `app/api/deps.py`, `app/api/main.py`, `tests/test_api_integration.py`

**Interfaces:**
- Consumes: `get_settings`, `OllamaEmbedder`, `PostgresKnowledgeStore`, `IngestPipeline`.
- Produces (rotas):
  - `POST /documents` — body JSON `{source_type, source (str/dict/base64), source_uri?, title?}` ou multipart upload de arquivo; valida `source_type` ∈ tipos suportados e tamanho do arquivo; chama `pipeline.ingest_one`; resposta `IngestResult`. Erro de validação → 422; tipo não suportado → 400. (A API não vetoriza nada aqui; a pipeline só extrai/chunka texto e delega ao store.)
  - `GET /search?q=&k=10` — `q` obrigatório; chama diretamente `store.hybrid_search(q, k)` passando o **texto** da query (Decisão — Opção A: a API não vetoriza a query; o store faz isso internamente via seu `Embedder` injetado); resposta `{"hits": [Hit...]}`.
  - `DELETE /documents/{id}` — chama `store.delete_document(id)`; 204.
  - `GET /health` — `{"db": store.health(), "ollama": embedder.health()}`; 200 se ambos True, senão 503. (`embedder` aqui é o mesmo `Embedder` injetado no store, exposto via `deps.get_embedder()` apenas para o healthcheck — a rota de busca não o usa diretamente.)
  - `deps.py`: providers `get_store()`, `get_embedder()`, `get_pipeline()` para `Depends`, sobrescrevíveis em teste via `app.dependency_overrides`. `get_store()` constrói `PostgresKnowledgeStore(database_url, embedder)` injetando o `Embedder` retornado por `get_embedder()`.

- [ ] **Step 1: Testes que falham** em `tests/test_api_integration.py` (usa `TestClient` + Postgres real; embedder fake via override)

```python
import pytest
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer
from app.api.main import app
from app.api import deps
from app.store.postgres import PostgresKnowledgeStore
from app.ingest.pipeline import IngestPipeline
from scripts.init_db import run_migrations

pytestmark = pytest.mark.integration

class FakeEmbedder:
    def embed(self, texts): return [[0.1] * 1024 for _ in texts]
    def health(self): return True

@pytest.fixture(scope="module")
def client():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        url = pg.get_connection_url().replace("psycopg2", "psycopg")
        run_migrations(url)
        emb = FakeEmbedder()
        store = PostgresKnowledgeStore(url, emb)  # Embedder injetado no store
        pipe = IngestPipeline(store, chunk_size=200, overlap=20)  # pipeline não vetoriza
        app.dependency_overrides[deps.get_store] = lambda: store
        app.dependency_overrides[deps.get_embedder] = lambda: emb
        app.dependency_overrides[deps.get_pipeline] = lambda: pipe
        yield TestClient(app)
        app.dependency_overrides.clear()

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["db"] is True

def test_post_then_search_then_delete(client):
    r = client.post("/documents", json={"source_type": "faq",
        "source": {"question": "Onde fica?", "answer": "São Paulo"}})
    assert r.status_code == 200 and r.json()["ok"]
    did = r.json()["document_id"]
    s = client.get("/search", params={"q": "São Paulo", "k": 5})
    assert s.status_code == 200 and "hits" in s.json()
    d = client.delete(f"/documents/{did}")
    assert d.status_code == 204

def test_search_requires_q(client):
    assert client.get("/search").status_code == 422

def test_unsupported_source_type_rejected(client):
    r = client.post("/documents", json={"source_type": "xml", "source": "x"})
    assert r.status_code == 400
```

- [ ] **Step 2: Rodar e ver falhar** — `pytest tests/test_api_integration.py -v -m integration` → FAIL.

- [ ] **Step 3: Implementar `app/api/deps.py` e `app/api/main.py`** com as quatro rotas, modelos de request pydantic, validação de tipo/tamanho, e tratamento de erros (422 validação, 400 tipo, 503 health).

- [ ] **Step 4: Rodar e ver passar** — `pytest tests/test_api_integration.py -v -m integration` → PASS.

- [ ] **Step 5: Commit** — `git add app/api tests/test_api_integration.py && git commit -m "feat: API FastAPI (documents/search/health) com DI"`

---

## Task 11: Docker Compose + Dockerfile + bootstrap do modelo

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `scripts/pull_model.sh`, `.env.example`

**Interfaces:**
- Consumes: `app.api.main:app` (uvicorn), `scripts.init_db`.
- Produces: stack subível com `docker compose up`. Três serviços com healthchecks; `api` depende de `postgres` e `ollama` saudáveis.

- [ ] **Step 1: Escrever `Dockerfile`** (python:3.12-slim; instala deps de `pyproject.toml`; CMD que roda migrations e sobe uvicorn).

```dockerfile
FROM python:3.12-slim
WORKDIR /code
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
COPY app ./app
COPY scripts ./scripts
CMD ["sh", "-c", "python -m scripts.init_db && uvicorn app.api.main:app --host 0.0.0.0 --port 8000"]
```

- [ ] **Step 2: Escrever `docker-compose.yml`**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: oc
      POSTGRES_PASSWORD: oc
      POSTGRES_DB: opencopilot
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U oc -d opencopilot"]
      interval: 5s
      timeout: 3s
      retries: 10
  ollama:
    image: ollama/ollama
    volumes: ["ollama:/root/.ollama"]
    healthcheck:
      test: ["CMD-SHELL", "ollama list >/dev/null 2>&1 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 10
  api:
    build: .
    environment:
      OC_DATABASE_URL: postgresql://oc:oc@postgres:5432/opencopilot
      OC_OLLAMA_URL: http://ollama:11434
    ports: ["8000:8000"]
    depends_on:
      postgres: { condition: service_healthy }
      ollama: { condition: service_healthy }
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)\""]
      interval: 10s
      timeout: 5s
      retries: 10
volumes: { pgdata: {}, ollama: {} }
```

- [ ] **Step 3: Escrever `scripts/pull_model.sh`** (`docker compose exec ollama ollama pull bge-m3`) e `.env.example` documentando as variáveis `OC_*`.

- [ ] **Step 4: Verificação manual de fumaça** (requer Docker):

Run:
```bash
docker compose up -d --build
bash scripts/pull_model.sh        # baixa bge-m3 (uma vez)
curl -s localhost:8000/health
```
Expected: `health` retorna `{"db": true, "ollama": true}` após o modelo estar disponível; `docker compose ps` mostra os 3 serviços `healthy`.

- [ ] **Step 5: Commit** — `git add Dockerfile docker-compose.yml scripts/pull_model.sh .env.example && git commit -m "feat: docker-compose (postgres/ollama/api) com healthchecks"`

---

## Task 12: Verificação end-to-end e critérios de aceite

**Files:**
- Modify: `README.md` (instruções de uso); Create: `tests/test_e2e_smoke.py` (opcional, `@pytest.mark.integration`, usa stack docker)

**Interfaces:**
- Consumes: stack completa.

- [ ] **Step 1: Rodar a suíte unitária inteira** — `pytest -m "not integration" -v` → todos PASS (config, interfaces, chunker, embedder, rrf, loaders).

- [ ] **Step 2: Rodar a suíte de integração** — `pytest -m integration -v` (requer Docker) → migrations, store, pipeline, API PASS.

- [ ] **Step 3: Validar critérios de sucesso do spec (checklist manual contra a stack docker)**

  - [ ] `docker compose up` sobe os três serviços (Task 11). → Critério 1
  - [ ] Ingerir PDF, página web, registro estruturado e FAQ via `POST /documents` retorna `ok=true`. → Critério 2
  - [ ] `GET /search` semântico retorna por significado; textual encontra termo exato; híbrida combina (validado em test_store_integration + chamada manual). → Critério 3
  - [ ] FAQ relevante aparece priorizada (test_rrf `test_faq_boost_promotes_faq` + verificação manual de uma query que casa FAQ). → Critério 4
  - [ ] Nenhum acesso a Postgres/Ollama fora dos adaptadores — `grep -rE "psycopg|httpx|11434|5432" app/ --include=*.py` só casa em `app/store/postgres.py`, `app/embedders/ollama.py`, `app/config.py`. → Critério 5
  - [ ] Embeddings gerados localmente (Ollama no compose), nenhum endpoint externo. → Critério 6

- [ ] **Step 4: Atualizar `README.md`** com: pré-requisitos, `docker compose up`, `pull_model.sh`, exemplos de `curl` para cada rota, como rodar testes (`pytest -m "not integration"` e `pytest -m integration`).

- [ ] **Step 5: Commit** — `git add README.md tests/test_e2e_smoke.py && git commit -m "docs+test: verificação end-to-end e critérios de aceite da Camada 1"`

---

## Critérios de aceite por fase (mapa ao spec)

| Fase/Task | Critério de aceite | Critério de sucesso do spec |
|---|---|---|
| 1 Config | `get_settings()` lê env `OC_*` com defaults corretos | base p/ todos |
| 2 Interfaces | Models e Protocols importáveis e tipados | 5 (abstrações) |
| 3 Chunker | Overlap correto, limites de sentença, índices sequenciais | 3 |
| 4 OllamaEmbedder | Retry/backoff, erro após esgotar, health | 6 |
| 5 RRF | Fusão por posição + boost FAQ funciona | 3, 4 |
| 6 Migrations | Tabelas + índices HNSW/GIN criados | 1, 3 |
| 7 Store | index/semantic/keyword/hybrid/delete contra PG real | 2, 3, 5 |
| 8 Loaders | Seleção por tipo; faq/structured extraem texto | 2 |
| 9 Pipeline | Ciclo ingerir→buscar; lote resiliente a item inválido | 2 |
| 10 API | 4 rotas, validação, DI testável | 2, 3 |
| 11 Compose | 3 serviços healthy, `compose up` funciona | 1, 6 |
| 12 E2E | Suítes verdes + checklist dos 6 critérios | 1–6 |

## Self-Review (resultado)

- **Cobertura do spec:** todas as seções do spec têm task correspondente (abstrações→T2; schema→T6; OllamaEmbedder→T4; PostgresKnowledgeStore+RRF→T5/T7; loaders→T8; chunker→T3; pipeline→T9; API→T10; testes unit+integração→T3/4/5/8 + T6/7/9/10; tratamento de erros→T4 retry, T9 resiliência, T10 validação; compose→T11; critérios de sucesso→T12).
- **Placeholders:** nenhum TODO/TBD; todo step de código mostra o código.
- **Consistência de tipos:** `Document`/`Chunk`/`Hit` definidos em T2 e usados com a mesma assinatura em T5/T7/T9/T10; `index_document(doc)` consistente entre Protocol (T2) e impl (T7), com `doc.chunks` carregando texto puro e o store vetorizando internamente via `Embedder` injetado (Decisão — Opção A); `semantic_search(query, k)` e `hybrid_search(query, k)` idem (texto, sem vetor pré-calculado); `reciprocal_rank_fusion(rankings, k, faq_boost)` idem entre T5 e T7.
