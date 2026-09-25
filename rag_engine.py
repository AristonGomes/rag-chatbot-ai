import os
import glob
import fitz  # PyMuPDF
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.3"))

# Variáveis globais para o índice, modelo e metadados dos chunks textuais
modelo_embeddings = None
index_faiss = None
document_chunks = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks_metadata.json")

def _inicializar_modelo():
    """ Inicializa o modelo (all-MiniLM-L6-v2) apenas quando necessário. """
    global modelo_embeddings
    if modelo_embeddings is None:
        modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

def carregar_indice():
    """ Carrega do disco o banco de dados vetorial caso exista. """
    global index_faiss, document_chunks
    if index_faiss is not None:
        return
        
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        try:
            index_faiss = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                document_chunks = json.load(f)
            print("Índice FAISS armazenado em disco foi recarregado preventivamente!")
        except Exception as e:
            print(f"Não foi possível aproveitar o índice em disco: {e}")
            index_faiss = None
            document_chunks = []

def salvar_indice():
    """ Comita blocos de texto e vetores no disco da máquina original. """
    global index_faiss, document_chunks
    if index_faiss is not None:
        faiss.write_index(index_faiss, FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(document_chunks, f, ensure_ascii=False)

def carregar_documento(caminho: str) -> str:
    """ Extrai de PDF e TXT mantendo compatibilidade nativa """
    texto = ""
    ext = os.path.splitext(caminho)[1].lower()
    
    try:
        if ext == ".pdf":
            doc = fitz.open(caminho)
            for page in doc:
                texto += page.get_text("text") + "\n"
            doc.close()
        elif ext == ".txt":
            with open(caminho, "r", encoding="utf-8") as f:
                texto = f.read()
    except Exception as e:
        print(f"Erro ao ler o documento {caminho}: {e}")
        
    return texto

def criar_chunks(texto: str, tamanho: int = 500, overlap: int = 50) -> list[str]:
    """ Quebra inteligente sem perder rastreio numérico do fim da sequência """
    chunks = []
    if not texto: return chunks
        
    inicio = 0
    tamanho_texto = len(texto)
    
    while inicio < tamanho_texto:
        fim = inicio + tamanho
        chunks.append(texto[inicio:fim])
        inicio = fim - overlap
        
    return chunks

def indexar_documentos(pasta: str = "documents/") -> dict:
    """
    Realiza o indexamento do Zero, mapeando o score de similaridade global (Cosine),
    nome dos arquivos e armazenando tudo num arquivo serializado.
    """
    global index_faiss, document_chunks
    
    _inicializar_modelo()
    
    document_chunks = []
    index_faiss = None
    arquivos_lidos = 0
    total_de_chunks = 0
    
    if not os.path.exists(pasta):
        if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
        if os.path.exists(CHUNKS_PATH): os.remove(CHUNKS_PATH)
        return {"arquivos": 0, "total_chunks": 0}
        
    caminhos = glob.glob(os.path.join(pasta, "*.pdf")) + glob.glob(os.path.join(pasta, "*.txt"))
    
    for caminho in caminhos:
        texto = carregar_documento(caminho)
        if not texto.strip(): continue
            
        arquivos_lidos += 1
        nome_doc = os.path.basename(caminho)
        chunks_doc = criar_chunks(texto)
        
        # Mapeando origens nas tabelas
        for c in chunks_doc:
            document_chunks.append({
                "arquivo": nome_doc,
                "texto": c
            })
        total_de_chunks += len(chunks_doc)
        
    if document_chunks:
        textos = [d['texto'] for d in document_chunks]
        
        # Normalize gera vetores l2 nativos e nos permite usar Product para simular Cosseno
        embeddings = modelo_embeddings.encode(textos, normalize_embeddings=True)
        dimensao = embeddings.shape[1]
        
        # Index Flat por Similaridade de Cosseno no Numpy/FAISS
        index_faiss = faiss.IndexFlatIP(dimensao)
        embeddings_float32 = np.array(embeddings).astype("float32")
        index_faiss.add(embeddings_float32)
        salvar_indice()
    else:
        # Repositorio vazio deleta backup
        if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
        if os.path.exists(CHUNKS_PATH): os.remove(CHUNKS_PATH)
        
    return {"arquivos": arquivos_lidos, "total_chunks": total_de_chunks}

def buscar_contexto(pergunta: str, top_k: int = 4) -> list[dict]:
    """ Busca vetorial baseando em similaridade > MIN_SCORE """
    global index_faiss, document_chunks
    
    carregar_indice()
    
    if index_faiss is None or index_faiss.ntotal == 0:
        return []
        
    _inicializar_modelo()
    
    # Prepara consulta garantindo mesma normalização pra comparar alinhamentos
    pergunta_embedding = modelo_embeddings.encode([pergunta], normalize_embeddings=True)
    pergunta_embedding_np = np.array(pergunta_embedding).astype("float32")
    
    # Coletando pesos da proximidade vetorial na vizinhança top_k
    scores, indices = index_faiss.search(pergunta_embedding_np, top_k)
    
    resultados = []
    
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(document_chunks):
            # Adicional para filtro de similaridade e relevância purgada caso MIN_SCORE
            if float(score) >= MIN_SCORE:
                resultados.append({
                    "score": float(score),
                    "arquivo": document_chunks[idx]["arquivo"],
                    "texto": document_chunks[idx]["texto"]
                })
            
    return resultados

def get_index_stats() -> dict:
    global index_faiss
    carregar_indice()
    total = index_faiss.ntotal if index_faiss is not None else 0
    return {
        "index_ativo": index_faiss is not None and total > 0,
        "total_chunks_indexados": total
    }
