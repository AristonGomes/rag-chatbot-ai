import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Dicionário em memória para armazenar as instâncias de ChatSession
sessoes_ativas = {}

class ChatSession:
    def __init__(self):
        self.historico = []

    def adicionar_mensagem(self, role: str, content: str):
        if role not in ["user", "model"]:
            raise ValueError("O role deve ser 'user' ou 'model'.")
        self.historico.append({"role": role, "parts": [content]})

    def limpar_historico(self):
        self.historico = []

    def get_historico(self, max_mensagens: int = 10) -> list:
        """ Limite imposto pra manter as ultimas 10 mensagens apenas """
        return self.historico[-max_mensagens:]

def responder(pergunta: str, session_id: str, contexto_chunks: list[dict]) -> dict:
    if session_id not in sessoes_ativas:
        sessoes_ativas[session_id] = ChatSession()
        
    session = sessoes_ativas[session_id]
    
    # Lógica aprimorada contendo o rastreio base de qual arquivo enviamos ao gemini
    trechos_instrucao = []
    fontes_utilizadas = []
    
    for c in contexto_chunks:
        trechos_instrucao.append(f"Arquivo: {c['arquivo']}\nTrecho: {c['texto']}")
        fontes_utilizadas.append({
            "arquivo": c['arquivo'],
            "trecho_resumo": c['texto'][:50] + "..."
        })
        
    chunks_unidos = "\n\n---\n\n".join(trechos_instrucao)
    
    system_instruction = f"""Você é um assistente especialista nos documentos fornecidos. Responda APENAS com base no contexto abaixo. Se não souber, diga que a informação não está nos documentos.
         
Contexto:
{chunks_unidos}"""

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction
        )
        
        chat = model.start_chat(history=session.get_historico(max_mensagens=10))
        response = chat.send_message(pergunta)
        resposta_gerada = response.text
        
        session.adicionar_mensagem("user", pergunta)
        session.adicionar_mensagem("model", resposta_gerada)
        
        # Gestão de Token e Custo
        tokens_in = 0
        tokens_out = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            tokens_in = getattr(response.usage_metadata, "prompt_token_count", 0)
            tokens_out = getattr(response.usage_metadata, "candidates_token_count", 0)
        
        custo_usd = (tokens_in / 1_000_000.0 * 0.075) + (tokens_out / 1_000_000.0 * 0.30)
            
        return {
            "resposta": resposta_gerada,
            "session_id": session_id,
            "modelo_usado": MODEL_NAME,
            "tokens_entrada": tokens_in,
            "tokens_saida": tokens_out,
            "custo_estimado_usd": custo_usd,
            "fontes": fontes_utilizadas,
            "chunks_usados": len(contexto_chunks)
        }
        
    except ResourceExhausted:
        return {
            "resposta": "Desculpe, a cota da API (ResourceExhausted) foi excedida. Aguarde alguns instantes.",
            "session_id": session_id,
            "modelo_usado": MODEL_NAME,
            "tokens_entrada": 0, "tokens_saida": 0, "custo_estimado_usd": 0.0,
            "fontes": fontes_utilizadas, "chunks_usados": len(contexto_chunks),
            "erro": "ResourceExhausted"
        }
    except Exception as e:
        return {
            "resposta": f"Falha interna comunicando com a API: {str(e)}",
            "session_id": session_id,
            "modelo_usado": MODEL_NAME,
            "tokens_entrada": 0, "tokens_saida": 0, "custo_estimado_usd": 0.0,
            "fontes": fontes_utilizadas, "chunks_usados": len(contexto_chunks),
            "erro": str(e)
        }

def listar_sessoes() -> list:
    return [{"session_id": sid, "tamanho_historico": len(sess.historico)} for sid, sess in sessoes_ativas.items()]
