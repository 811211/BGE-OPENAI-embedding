# ===========================================
# 📄 QA_Embedding
# 功能：生成問題並比較模型
# ===========================================
import json
import os
import psycopg2
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict
from openai import AzureOpenAI
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from FlagEmbedding import BGEM3FlagModel

# ----- 環境配置 -----
load_dotenv()

print("DEBUG: AOAI_KEY =", os.getenv("AOAI_KEY"))
print("DEBUG: AOAI_ENDPOINT =", os.getenv("AOAI_ENDPOINT"))
print("DEBUG: AOAI_API_VERSION =", os.getenv("AOAI_API_VERSION"))
print("DEBUG: AOAI_CHAT_DEPLOYMENT =", os.getenv("AOAI_CHAT_DEPLOYMENT"))

# 初始化 AzureOpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AOAI_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_version=os.getenv("AOAI_API_VERSION")
)

embedding_ada = AzureOpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    azure_endpoint=os.getenv("EMBEDDING_URL"),
    api_version=os.getenv("AOAI_API_VERSION"),
)

BGE_MODEL = BGEM3FlagModel(
    r"C:\Users\user\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181",
    use_fp16=True
)

# DB config
DB_CONFIG = {
    "host": "210.67.12.103",
    "database": "Learning",
    "user": "editor",
    "password": "systex.6214",
    "port": 65432
}

# ----- 抽取資料庫文件 -----
def fetch_documents(limit=1000) -> List[Dict]:
    max_chars = 2000
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute('SELECT DISTINCT source_table FROM "2500567RAG2";')
    source_types = [row[0] for row in cur.fetchall()]
    
    result = {}

    for s_type in source_types:
        cur.execute('SELECT document_id, text FROM "2500567RAG2" WHERE source_table = %s LIMIT %s;', (s_type, limit))
        docs = []
        for row in cur.fetchall():
            doc_id, text = row
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()[:max_chars]
            docs.append({"document_id": doc_id, "text": text, "source_table": s_type})
        result[s_type] = docs
    
    conn.close()
    return result

# ----- 生成假問題 -----
def generate_questions_for_docs(docs: List[Dict], total_questions=100) -> List[Dict]:
    chat_deployment = os.getenv("AOAI_CHAT_DEPLOYMENT")
    questions = []
    existing_questions = set()  # ➤ 用於避免重複問題
    num_docs = len(docs)
    if num_docs == 0:
        return []

    questions_per_doc = total_questions // num_docs
    extra = total_questions % num_docs

    for idx, doc in enumerate(tqdm(docs, desc="生成假問題中")):
        n = questions_per_doc + (1 if idx < extra else 0)

        for _ in range(n):
            retry = 0
            max_retry = 3  # 最多重試次數

            while retry < max_retry:
                prompt = (
                    f"請針對以下內容，提出一個可以直接從內文找到答案的測試問題，並包含『能唯一指涉答案的關鍵字』，"
                    f"例如數值、公司名、年份或具體事件等，使問題明確對應單一答案。\n"
                    f"要求：\n"
                    f"1️⃣ 問題僅根據內文，不做額外推測。\n"
                    f"2️⃣ 問題簡短具體。\n"
                    f"3️⃣ 使用繁體中文。\n"
                    f"4️⃣ 僅輸出問題內容，不要加「問題：」或其他說明。\n\n"
                    f"範例：\n"
                    f"❌『這家公司營收如何？』（不具唯一性）\n"
                    f"✅『2023 年該公司營收為多少？』（具體、明確）\n\n"
                    f"內文：\n{doc['text']}"
                )

                try:
                    response = client.chat.completions.create(
                        model=chat_deployment,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        max_tokens=100
                    )
                    q = response.choices[0].message.content.strip()
                    a = extract_answer_from_text(doc["text"], q, client, chat_deployment)

                    if q in existing_questions:
                        retry += 1
                        continue  # 重複問題就重試
                    else:
                        existing_questions.add(q)
                        questions.append({
                            "document_id": doc["document_id"],
                            "question": q,
                            "answer": a
                        })
                        break  # 成功生成問題就跳出 retry 迴圈

                except Exception as e:
                    print(f"⚠️ 生成假問題失敗 document_id={doc['document_id']}，原因: {e}")
                    break  # 若是 LLM API 失敗就跳出 retry
    return questions

# ----- 從text擷取answer -----

def extract_answer_from_text(text: str, question: str, client=None, deployment=None) -> str:
    prompt = (
    f"""根據以下內容與問題，請從內文中擷取**最精準的一段文字**作為答案，不要自行改寫或補充，\n
    答案必須是原文中的一段話。\n
    答案格式必須為：**「關鍵詞:數字」**（例如：股價變化:500、營收:1200、淨利:80 等）。\n
    且必須遵循以下json格式:
    
    {text}
        
    【問題】
    {question}

    【輸出】
    {{
    "answer": "股價變化:500"
    }}
    """
    )
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100,
            response_format={ "type": "json_object" }
        )
        
        
        json_data = json.loads(response.choices[0].message.content.strip())
        return json_data.get("answer", "")
    except Exception as e:
        print(f"⚠️ 擷取答案失敗：{e}")
        return ""  # or return None



# ----- embedding 函數 -----

def embed_openai(text: str) -> np.ndarray:
    try:
        response = embedding_ada.embeddings.create(
            input=text,
            model=os.getenv("EMBEDDING_MODEL")
        )
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        print(f"embed_openai 出錯: {e}")
        return np.zeros((1536,), dtype="float32")

def embed_bge(text: str) -> np.ndarray:
    try:
        output = BGE_MODEL.encode(text)
        dense_vecs = output.get("dense_vecs")
        if dense_vecs is None:
            return np.zeros((1024,), dtype="float32")
        return dense_vecs.astype("float32")
    except Exception as e:
        print(f"embed_bge 出錯: {e}")
        return np.zeros((1024,), dtype="float32")
    
# ----- 資料儲存 -----

def save_question(entry: Dict):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO "2500567RAG" (document_id, source_table, question, answer, question_embedding_bge, question_embedding_openai, answer_embedding_bge, answer_embedding_openai, text) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);',
        (
            entry["document_id"],
            entry["source_table"],
            entry["question"],
            entry["answer"],
            entry["q_emb_bge"].tolist(),
            entry["q_emb_openai"].tolist(),
            entry["a_emb_bge"].tolist(),
            entry["a_emb_openai"].tolist(),
            entry["text"]
        )
    )
    conn.commit()
    conn.close()

def check_and_clear_table_if_needed():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM public."2500567RAG"')
        existing_count = cur.fetchone()[0]
        
        if existing_count > 0:
            print(f"⚠️ 資料庫已有 {existing_count} 筆記錄")
            resp = input("請選擇動作：\n"
                        "y = 刪除資料並重新處理\n"
                        "n = 保留資料並繼續處理\n"
                        "exit = 離開程式\n"
                        "輸入選項 (y/n/exit): ").strip().lower()
            
            if resp.lower() == 'y':
                cur.execute('DELETE FROM public."2500567RAG"')
                cur.execute('TRUNCATE TABLE public."2500567RAG" RESTART IDENTITY;')
                conn.commit()
                print("✅ 已清除資料，重新開始處理")
            elif resp == 'n':
                print("✅ 保留現有資料，繼續處理")
            elif resp == 'exit':
                print("🚪 已取消處理，結束程式")
                cur.close()
                conn.close()
                return False
            else:
                print("⚠️ 無效選項，請重新執行並輸入 y/n/exit")
                cur.close()
                conn.close()
                return False
        cur.close()
        conn.close()
        return True
    
    except Exception as e:
        print(f"檢查/清除資料失敗: {e}")
        return False



# ----- 主程式 -----

def main():
    proceed = check_and_clear_table_if_needed()
    if not proceed:
        return

# =============開始抽取資料=============
    print("開始依照 source_table 類型抽取資料...")
    grouped_docs = fetch_documents(limit=1000)

    total_all_questions = 0
    
    for source_table, docs in grouped_docs.items():
        print(f"\n🗂️ 類型: {source_table}，共 {len(docs)} 筆")
        print("🔄 開始生成問題...")
        
        questions = generate_questions_for_docs(docs, total_questions=100)
        print(f"✅ 共為類型 {source_table} 生成 {len(questions)} 筆問題")

        print("💾 生成答案並寫入資料庫...")
        for q in tqdm(questions, desc=f"{source_table} - 寫入中"):
            doc = next((d for d in docs if d["document_id"] == q["document_id"]), None)
            if doc:
                
                # 擷取答案
                answer = answer = extract_answer_from_text(doc["text"], q["question"], client, os.getenv("AOAI_CHAT_DEPLOYMENT"))
                
                # 計算向量
                q_emb_bge = embed_bge(q["question"])
                q_emb_openai = embed_openai(q["question"])
                a_emb_bge = embed_bge(doc["text"])
                a_emb_openai = embed_openai(doc["text"])
                
                # 儲存資料
                entry = {
                    "document_id": doc["document_id"],
                    "source_table": doc["source_table"],
                    "question": q["question"],
                    "answer": answer,
                    "q_emb_bge": q_emb_bge,
                    "q_emb_openai": q_emb_openai,
                    "a_emb_bge": a_emb_bge,
                    "a_emb_openai": a_emb_openai,
                    "text": doc["text"]
                }
                save_question(entry)
                
        total_all_questions += len(questions)

    print(f"\n✅ 全部類型問題與答案生成完畢！總數: {total_all_questions} 筆")



if __name__ == "__main__":
    extract_answer_from_text("測試問題", "來源名稱", client, os.getenv("AOAI_CHAT_DEPLOYMENT"))
    main()
