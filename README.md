# RAG Chatbot AI 🤖

Um Chatbot inteligente que utiliza a técnica de **RAG (Retrieval-Augmented Generation)** para responder perguntas baseadas em seus próprios documentos locais. O sistema utiliza a API do **Google Gemini** para geração de texto e **FAISS** com **Sentence Transformers** para a busca semântica.

## 🚀 Funcionalidades

- **Processamento de Documentos**: Suporte para arquivos `.pdf` e `.txt`.
- **Busca Semântica Avançada**: Utiliza embeddings vetoriais para encontrar os trechos mais relevantes.
- **Persistent Index**: O banco de dados vetorial (FAISS) é salvo em disco, evitando a reindexação a cada reinício.
- **Interface Moderna**: UI estilo ChatGPT com suporte a Markdown, Dark Mode e animações.
- **Rastreabilidade**: Informa quais fontes e trechos foram usados para gerar cada resposta.
- **Gestão de Custos**: Estimativa de custo (em USD) baseada nos tokens de entrada e saída da API.
- **Performance e Segurança**: Rate limiting de requisições e persistência de histórico (limitado a 10 mensagens por sessão).

## 🛠️ Tecnologias Utilizadas

- **Backend**: Python 3.9+, FastAPI, Uvicorn.
- **IA/ML**: Google Generative AI (Gemini 1.5 Flash), Sentence-Transformers (`all-MiniLM-L6-v2`), FAISS-CPU.
- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript (Vanilla), Marked.js (Markdown).
- **Processamento de PDF**: PyMuPDF (fitz).

## 📋 Pré-requisitos

Antes de começar, você precisará de:
1. Uma chave de API do Google Gemini. Obtenha em: [Google AI Studio](https://aistudio.google.com/).
2. Python instalado (versão 3.9 ou superior recomendada).

## ⚙️ Configuração

1. **Clonar/Extrair o projeto**:
   ```bash
   cd rag-chatbot-ai
   ```

2. **Instalar Dependências**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar Variáveis de Ambiente**:
   - Renomeie o arquivo `.env.example` para `.env`.
   - Insira sua chave da API:
     ```env
     GEMINI_API_KEY=SUA_CHAVE_AQUI
     GEMINI_MODEL=gemini-1.5-flash
     MAX_CHUNKS=4
     MIN_SCORE=0.3
     ```

## 🏃 Como Rodar

1. **Inicie o servidor (Backend)**:
   ```bash
   uvicorn main:app --reload
   ```

2. **Acesse a Aplicação**:
   - Abra o navegador em `http://127.0.0.1:8000`.
   - A interface do chatbot será carregada automaticamente.

## 📂 Estrutura do Projeto

```text
rag-chatbot-ai/
├── backend/
│   ├── main.py              # Servidor FastAPI e Endpoints
│   ├── rag_engine.py        # Processamento de documentos e FAISS
│   └── chat_handler.py      # Integração com Gemini e Histórico
├── frontend/
│   └── index.html           # Interface do Usuário (Single Page)
├── documents/               # Pasta de upload dos arquivos
├── .env                     # Configurações sensíveis (não commitado)
├── requirements.txt         # Lista de bibliotecas
└── faiss_index.bin          # Índice vetorial persistido (gerado no primeiro uso)
```

## 🔌 API Endpoints Principais

- `POST /upload`: Envia um novo documento para indexação.
- `POST /chat`: Envia uma pergunta e recebe a resposta contextualizada.
- `GET /documentos`: Lista os arquivos atualmente no banco de dados.
- `DELETE /documentos/{nome}`: Remove um documento e atualiza o índice.
- `GET /status`: Mostra dados técnicos do sistema (chunks, documentos, etc).

## ⚠️ Notas Importantes

- **Rate Limit**: O sistema permite até 20 requisições por minuto por IP para as rotas de chat e upload.
- **Memória**: O histórico de cada sessão é mantido em memória no servidor e limitado às últimas 10 trocas de mensagens.
- **Indexação**: Ao deletar ou adicionar arquivos, o sistema reconstrói o índice `faiss_index.bin` automaticamente para garantir a precisão.

---
Desenvolvido como um projeto de demonstração de arquitetura RAG com Gemini.
