# Opção de implantação em produção: Cloudflare

**Data da pesquisa:** 2026-06-24
**Status:** Avaliado — **não adotado para o MVP**; registrado como opção futura.

## Decisão

O MVP da Camada 1 segue **self-hosted em Docker** (Postgres + pgvector, Ollama +
bge-m3, FastAPI), agnóstico de host. A Cloudflare foi avaliada como alvo de
produção e **não foi adotada agora**, por colidir com dois requisitos que
motivaram o projeto. Este documento preserva a análise para uma decisão futura.

## Por que não agora — os dois requisitos críticos

1. **Evitar lock-in.** Os serviços gerenciados nativos da Cloudflare (Vectorize,
   Workers AI, D1, AI Search) são proprietários, sem protocolo padrão. Adotá-los
   troca o lock-in da Microsoft pelo lock-in da Cloudflare.
2. **Dados não saírem do ambiente.** Embeddings locais (Ollama) foram uma escolha
   deliberada. No Workers AI, o texto dos documentos é enviado à infraestrutura
   da Cloudflare para inferência — sai da rede do usuário (mesmo a Cloudflare
   declarando que não treina modelos com esses dados).

Qualquer cloud, por definição, tira os dados do ambiente do usuário. Por isso o
alvo mais alinhado ao projeto continua sendo **uma VPS / servidor próprio**.

## Mapeamento dos componentes (estado em jun/2026)

| Nosso componente | Equivalente Cloudflare | Estado | Encaixa? |
|---|---|---|---|
| FastAPI (servidor contínuo) | Python Workers | open beta | ❌ modelo request/response, não roda FastAPI como servidor long-running |
| FastAPI / containers | **Cloudflare Containers** | **GA (13/04/2026)** | ✅ imagem Docker arbitrária, até 4 vCPU / 12 GiB / 20 GB disco |
| PostgreSQL + pgvector | Containers (Postgres próprio) ou Vectorize + D1 | GA | ⚠️ sem Postgres gerenciado nem pgvector nativo |
| pgvector (busca vetorial) | **Vectorize** | GA | ⚠️ proprietário; aceita até 1536-d (1024 do bge-m3 cabe) |
| RRF / híbrido (vetor + BM25) | **AI Search** (ex-AutoRAG) | hybrid/RRF desde 16/04/2026 | ⚠️ acoplado ao produto AI Search |
| Ollama + bge-m3 (1024-d, local) | Workers AI `@cf/baai/bge-m3` | GA, 1024-d | ⚠️ mesmo modelo, mas roda na GPU da Cloudflare (dados saem) |
| Ingestão (pypdf, python-docx, trafilatura) | Containers, ou parsing automático do AI Search | GA / beta | ⚠️ AI Search faz chunking próprio (menos controle fino) |
| Arquivos originais | **R2** | GA | ✅ free tier 10 GB, sem egress |
| Metadados relacionais | **D1** (SQLite serverless) | GA | ✅ para metadados — **não para vetores** |

## Dois cenários avaliados

### (A) Rodar o stack Docker atual em Cloudflare Containers
- **Viável:** sim. Containers GA suporta Docker arbitrário e processos longos.
- **Esforço:** médio. Sem `docker-compose.yml` nativo — topologia recriada via
  Worker/binding (classe `Container`). Validar RAM do Ollama+bge-m3 (aperta nos
  12 GiB) e **persistência de disco do Postgres entre reinícios** (não
  documentada — verificar antes).
- **Perde dos requisitos:** sem lock-in adicional de API (mesmo Docker), mas
  dados ainda passam pela nuvem da Cloudflare. Produto jovem (~2 meses de GA):
  exige piloto antes de produção.

### (B) Rearquitetar Cloudflare-native (Vectorize + Workers AI + R2 + Workers)
- **Viável:** sim, e elegante — bge-m3 gerenciado com 1024-d, Vectorize, AI
  Search com RRF, R2 para PDFs, D1 para metadados.
- **Esforço:** alto. Reescrita de arquitetura (Python Workers não substitui
  FastAPI 1:1; ingestão custom precisaria de Containers ou do parsing do AI
  Search).
- **Perde dos requisitos:** **quebra** o requisito de embeddings locais e
  **aumenta** o lock-in. Só faz sentido se a privacidade for relaxada.

## Recomendação

**Meio-termo recomendado para produção:** manter o backend self-hosted (Docker
numa VPS) e usar a Cloudflare **apenas na frente, como CDN + Tunnel** (TLS,
proteção DDoS, sem expor IP, cache). Ganha-se a borda da Cloudflare **sem**
enviar dados de documentos e **sem** lock-in no núcleo da busca.

A camada de abstração (`KnowledgeStore`, `Embedder`) preserva a porta de saída
para o cenário (B) no futuro, caso o requisito de privacidade afrouxe — bastaria
escrever um `VectorizeKnowledgeStore` + `WorkersAIEmbedder` sem tocar no resto.

## Pontos a reconfirmar antes de qualquer adoção

- Persistência de volume/disco em Cloudflare Containers entre reinícios
  (`developers.cloudflare.com/containers/platform-details/`).
- Limites de RPS/upsert do Vectorize.
- Dimensões do bge-m3 na página oficial do Workers AI / Playground (1024
  confirmado por fontes secundárias consistentes).

## Fontes principais

- Containers GA: https://developers.cloudflare.com/changelog/post/2026-04-13-containers-sandbox-ga/
- Containers preço/limites: https://developers.cloudflare.com/containers/pricing/
- Python Workers (open beta): https://developers.cloudflare.com/workers/languages/python/
- Vectorize limites/preço: https://developers.cloudflare.com/vectorize/platform/limits/
- Workers AI modelos: https://developers.cloudflare.com/workers-ai/models/
- bge-m3: https://developers.cloudflare.com/workers-ai/models/bge-m3/
- Workers AI uso de dados: https://developers.cloudflare.com/workers-ai/platform/data-usage/
- AI Search (ex-AutoRAG) + RRF: https://developers.cloudflare.com/changelog/post/2026-04-16-hybrid-search-and-relevance-boosting/
- R2 preço: https://developers.cloudflare.com/r2/pricing/
- D1 limites: https://developers.cloudflare.com/d1/platform/limits/
