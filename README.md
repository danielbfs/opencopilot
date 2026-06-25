# opencopilot

Alternativa self-hosted ao Microsoft Copilot Studio, construída para evitar
lock-in. O sistema é dividido em três camadas, construídas em ordem:

1. **Camada de conhecimento / busca semântica** — ingestão de documentos, busca
   híbrida (semântica + textual) sobre PostgreSQL + pgvector. *(em projeto)*
2. **Camada de RAG conversacional** — respostas fundamentadas geradas por LLM.
3. **Camada de plugins / ações** — chamadas a APIs e workflows.

Princípios: self-hosted (Docker), sem lock-in de banco ou cloud, dados não saem
do ambiente (embeddings gerados localmente).

## Documentação

- [Spec da Camada 1](docs/superpowers/specs/2026-06-24-camada1-busca-semantica-design.md)
- [Plano de implementação da Camada 1](docs/superpowers/plans/2026-06-24-camada1-busca-semantica-plan.md)
- [Opção de implantação em produção: Cloudflare](docs/deployment/2026-06-25-opcao-cloudflare.md)
