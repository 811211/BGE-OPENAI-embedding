# ===========================================
# 📄 demo2
# 功能：讀取固定問題集，生成 Embedding，並用 Faiss 對比模型效果
# ===========================================
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
    
# ----- 建立 FAISS 索引 -----

def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

# ----- 評估指標計算 -----

def evaluate_retrieval(index, query_vecs, true_indices, top_k=5):
    faiss.normalize_L2(query_vecs)
    D, I = index.search(query_vecs, top_k)

    total = len(true_indices)
    recall, precision, mrr, mean_rank, topk = 0, 0, 0, 0, 0

    for i in range(total):
        pred = I[i]
        ans = true_indices[i]

        if ans in pred:
            recall += 1
            rank = np.where(pred == ans)[0][0] + 1
            mean_rank += rank
            mrr += 1 / rank
            if rank == 1:
                topk += 1

        precision += int(ans in pred) / top_k

    return {
        "Recall@K": recall / total,
        "Precision@K": precision / total,
        "Top-K Accuracy": topk / total,
        "Mean Rank": mean_rank / total,
        "MRR": mrr / total
    }
    
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
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute('SELECT id, question, answer, question_embedding_bge, question_embedding_openai, answer_embedding_bge, answer_embedding_openai FROM "2500567RAG" ORDER BY id;')
    rows = cur.fetchall()

    bge_q_vecs, bge_a_vecs, openai_q_vecs, openai_a_vecs = [], [], [], []
    ground_truth = []  # 對應每一筆 question 的正解是第幾筆 answer

    for i, row in enumerate(rows):
        bge_q_vecs.append(row["question_embedding_bge"])
        bge_a_vecs.append(row["answer_embedding_bge"])
        openai_q_vecs.append(row["question_embedding_openai"])
        openai_a_vecs.append(row["answer_embedding_openai"])
        ground_truth.append(i)

    bge_q = np.array(bge_q_vecs, dtype='float32')
    bge_a = np.array(bge_a_vecs, dtype='float32')
    openai_q = np.array(openai_q_vecs, dtype='float32')
    openai_a = np.array(openai_a_vecs, dtype='float32')

    print("\n[🔍] 評估 BGE...")
    result_bge = evaluate_retrieval(build_faiss_index(bge_a), bge_q, ground_truth)
    for k, v in result_bge.items():
        print(f"{k}: {v:.4f}")

    print("\n[🔍] 評估 OpenAI...")
    result_openai = evaluate_retrieval(build_faiss_index(openai_a), openai_q, ground_truth)
    for k, v in result_openai.items():
        print(f"{k}: {v:.4f}")

    conn.close()

if __name__ == "__main__":
    main()
