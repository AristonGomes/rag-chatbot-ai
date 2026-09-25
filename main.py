import os
import uuid
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_engine
import chat_handler

app = FastAPI(title="RAG Chatbot AI")

# Banco rápido em memória para proteger portas API base contra Spam
rate_limit_db = {}
MAX_REQUESTS_PER_MIN = 20

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    # Protege com rate limit unicamente os consumos que disparam recursos, aliviando o frontend do app.
    if "/chat" in request.url.path or "/upload" in request.url.path:
        ip = request.client.host
        agora = time.time()
        
        if ip not in rate_limit_db:
            rate_limit_db[ip] = []
            
        rate_limit_db[ip] = [t for t in rate_limit_db[ip] if agora - t < 60.0]
        
        if len(rate_limit_db[ip]) >= MAX_REQUESTS_PER_MIN:
            return JSONResponse(status_code=429, content={"detail": f"Limite de spam excedido ({MAX_REQUESTS_PER_MIN} acessos/min). Tente novamente logo mais."})
            
        rate_limit_db[ip].append(agora)
        
    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Expandido ou mapeável para os domínos requeridos de front end
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
FRONTEND_PATH = os.path.join(BASE_DIR, "frontend", "index.html")

@app.on_event("startup")
async def startup_event():
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    stats = rag_engine.get_index_stats()
    # Verifica em tempo real, se existe index no disco, FAISS não reconstrói do zero.
    if not stats.get("index_ativo"):
        print("[Boot] Sem index prévio. Iniciando indexação do zero na pasta documents/...")
        rag_engine.indexar_documentos(DOCUMENTS_DIR)
    else:
        print("[Boot] Índice recuperado fisicamente com sucesso sem esforço computacional!")

@app.get("/")
async def frontend_view():
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH)
    raise HTTPException(status_code=404, detail="Index.html Frontend Desaparecido.")

@app.post("/upload")
async def upload_documento(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Apenas documentos em .PDF ou .TXT são permitidos!")
        
    caminho_arquivo = os.path.join(DOCUMENTS_DIR, file.filename)
    try:
        content = await file.read()
        with open(caminho_arquivo, "wb") as f:
            f.write(content)
            
        chunks_anteriores = rag_engine.get_index_stats().get("total_chunks_indexados", 0)
        stats = rag_engine.indexar_documentos(DOCUMENTS_DIR)
        chunks_novos = max(0, stats["total_chunks"] - chunks_anteriores)
        
        return {"arquivo": file.filename, "chunks_gerados": chunks_novos, "total_chunks": stats["total_chunks"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    pergunta: str
    session_id: Optional[str] = None

@app.post("/chat")
async def efetuar_chat(req: ChatRequest):
    session_uuid = req.session_id if req.session_id else str(uuid.uuid4())
    
    val_max_chunks = int(os.getenv("MAX_CHUNKS", "4"))
    
    chunks_contexto = rag_engine.buscar_contexto(req.pergunta, top_k=val_max_chunks)
    resultado_gemini = chat_handler.responder(req.pergunta, session_uuid, chunks_contexto)
    
    return resultado_gemini

@app.get("/chat/historico/{session_id}")
async def fetch_historico(session_id: str):
    if session_id in chat_handler.sessoes_ativas:
        # Traz as 10 últimas como requisitado no chat_handler
        sessao_memoria = chat_handler.sessoes_ativas[session_id].get_historico()
        return {"session_id": session_id, "historico": sessao_memoria}
    raise HTTPException(status_code=404, detail="Aquivo inexistente ou memória inativa do chatbot.")

@app.get("/documentos")
async def visualizar_documentos():
    dados = []
    if os.path.exists(DOCUMENTS_DIR):
        for f in os.listdir(DOCUMENTS_DIR):
            caminho_completo = os.path.join(DOCUMENTS_DIR, f)
            if os.path.isfile(caminho_completo):
                dados.append({"nome": f, "tamanho": os.path.getsize(caminho_completo)})
    return dados

@app.delete("/documentos/{nome}")
async def deletar_documento(nome: str):
    caminho_arquivo = os.path.join(DOCUMENTS_DIR, nome)
    if not os.path.exists(caminho_arquivo):
        raise HTTPException(status_code=404, detail="Arquivo inexistente.")
    
    os.remove(caminho_arquivo)
    stats = rag_engine.indexar_documentos(DOCUMENTS_DIR)
    return {"message": "Documento apagado.", "stats": stats}

@app.post("/chat/limpar/{session_id}")
async def reset_sessao(session_id: str):
    if session_id in chat_handler.sessoes_ativas:
        chat_handler.sessoes_ativas[session_id].limpar_historico()
        return {"status": "Histórico limpo"}
    raise HTTPException(status_code=404, detail="Sessão nunca acionada ou inexistente.")

@app.get("/status")
async def dashboard_status():
    index_meta = rag_engine.get_index_stats()
    sessoes_length = len(chat_handler.listar_sessoes())
    modelo = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    qtd_docs = len([a for a in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, a))]) if os.path.exists(DOCUMENTS_DIR) else 0
        
    return {
        "documentos_indexados": qtd_docs,
        "chunks": index_meta.get("total_chunks_indexados", 0),
        "sessoes_ativas": sessoes_length,
        "modelo_em_uso": modelo
    }
