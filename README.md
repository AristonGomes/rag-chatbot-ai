# RAG Chatbot AI

Aplicação de pergunta e resposta sobre documentos locais baseada em **RAG (Retrieval-Augmented Generation)**. O sistema combina indexação vetorial local com **FAISS** e **Sentence Transformers** para busca semântica e utiliza a API do **Google Gemini (`gemini-1.5-flash`)** para geração de respostas fundamentadas nas fontes.

---

## Funcionalidades

- **Ingestão de Documentos:** upload e fragmentação (*chunking*) de arquivos `.pdf` (via `PyMuPDF`) e `.txt`.
- **Busca Semântica Local:** geração de embeddings com o modelo `all-MiniLM-L6-v2` e armazenamento em índice vetorial `FAISS` persistido em disco (`faiss_index.bin`).
- **Rastreabilidade de Fontes:** cada resposta informa os documentos e trechos utilizados como contexto, reduzindo alucinações.
- **Estimativa de Custos e Tokens:** cálculo em tempo real do consumo de tokens de entrada/saída e custo estimado por interação.
- **Gestão de Sessão e Segurança:** histórico contextual em memória (limitado às últimas 10 interações por sessão) e *rate limiting* (20 requisições/minuto por IP nas rotas críticas).
- **Interface Web Integrada:** frontend single-page com suporte a Markdown, gerenciamento de arquivos indexados e indicadores técnicos da base vetorial.

---

## Tecnologias Utilizadas

- **Backend:** Python 3.9+, FastAPI, Uvicorn
- **Busca Vetorial & Embeddings:** `faiss-cpu`, `sentence-transformers` (`all-MiniLM-L6-v2`)
- **LLM:** Google Generative AI (`gemini-1.5-flash`)
- **Processamento de Documentos:** `PyMuPDF` (`fitz`)
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla) e `Marked.js`

---

## Estrutura do Projeto

```text
rag-chatbot-ai/
├── main.py              # Servidor FastAPI, rotas HTTP e rate limiting
├── rag_engine.py        # Extração de texto, chunking, embeddings e índice FAISS
├── chat_handler.py      # Integração com Google Gemini, histórico e cálculo de tokens
├── frontend/
│   └── index.html       # Interface web do chat e painel de documentos
├── documents/           # Diretório de armazenamento dos arquivos enviados
├── requirements.txt     # Dependências do projeto
└── .env.example         # Template de variáveis de ambiente
```

---

## Configuração e Execução Local

### 1. Ambiente Virtual e Dependências

```bash
# Criar e ativar ambiente virtual (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt
```

### 2. Variáveis de Ambiente

Copie o arquivo `.env.example` para `.env` e adicione sua chave do [Google AI Studio](https://aistudio.google.com/):

```env
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-1.5-flash
MAX_CHUNKS=4
MIN_SCORE=0.3
```

### 3. Iniciar o Servidor

```bash
uvicorn main:app --reload
```

Acesse a interface no navegador em `http://127.0.0.1:8000`.

---

## Endpoints da API

- **`POST /upload`** — Recebe um documento (`.pdf` ou `.txt`), gera os embeddings e atualiza o índice FAISS.
- **`POST /chat`** — Recebe a pergunta do usuário, recupera os trechos mais relevantes (`top-k`) e retorna a resposta gerada com as fontes e métricas de tokens.
- **`GET /documentos`** — Lista os documentos atualmente indexados.
- **`DELETE /documentos/{nome}`** — Remove um arquivo e reconstrói o índice vetorial.
- **`GET /status`** — Retorna estatísticas operacionais (total de documentos, chunks indexados e modelo ativo).

---

## Autor e Licença

- **Autor:** [Ariston Gomes](https://www.linkedin.com/in/ariston-gomes/)
- **Licença:** Distribuído sob a licença MIT.
