# ===========================================
# 📄 demo2
# 功能：讀取固定問題集，生成 Embedding，並用 Faiss 對比模型效果
# ===========================================

import os
import shutil
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
from typing import List, Dict

import numpy as np
import faiss
from tqdm import tqdm
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from openai import AzureOpenAI
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

# ----- 工具函數 -----
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

def evaluate_retrieval(index, queries, ground_truth, k=5):
    D, I = index.search(queries, k)
    recall_at_k = []
    precision_at_k = []
    top_k_accuracy = []
    mean_rank = []
    reciprocal_ranks = []

    for i, retrieved in enumerate(I):
        relevant = ground_truth[i]
        try:
            rank = list(retrieved).index(relevant)
            recall_at_k.append(1)
            precision_at_k.append(1 / (rank + 1))
            top_k_accuracy.append(1 if rank == 0 else 0)
            mean_rank.append(rank + 1)
            reciprocal_ranks.append(1 / (rank + 1))
        except ValueError:
            recall_at_k.append(0)
            precision_at_k.append(0)
            top_k_accuracy.append(0)
            mean_rank.append(k + 1)
            reciprocal_ranks.append(0)

    return {
        "Recall@K": round(np.mean(recall_at_k), 4),
        "Precision@K": round(np.mean(precision_at_k), 4),
        "Top-1 Accuracy": round(np.mean(top_k_accuracy), 4),
        "Mean Rank": round(np.mean(mean_rank), 4),
        "MRR": round(np.mean(reciprocal_ranks), 4),
    }

# ----- 資料儲存 -----

def save_evaluation_to_txt(results: dict, run_count: int):
    filename = f"BGE-OPENAI-embedding test({run_count}).txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"BGE-OPENAI Embedding Evaluation Report (Runs = {run_count})\n")
        f.write("=" * 60 + "\n\n")
        for source, models in results.items():
            f.write(f"📄 Source Table: {source}\n")
            f.write("-" * 60 + "\n")
            for model_name, metric_data in models.items():
                f.write(f"🔹 {model_name}:\n")
                for metric, values in metric_data.items():
                    if run_count == 1:
                        f.write(f"{metric}: {values[0]}\n")
                    else:
                        avg = round(sum(values) / len(values), 4)
                        f.write(f"{metric}: Run1={values[0]}  Run2={values[1]}  Run3={values[2]}  Avg={avg}\n")
                f.write("\n")
    return filename


def prompt_save_as(src_path):
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(
        title="另存結果為",
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt")]
    )
    if save_path:
        shutil.copyfile(src_path, save_path)
        print(f"✅ 成功另存為：{save_path}")
    else:
        print("❌ 取消另存")




def generate_analysis(results_dict):
    import json
    return analyze_results_with_llm(json.dumps(results_dict, ensure_ascii=False, indent=2))



def analyze_results_with_llm(text: str) -> str:
    analysis_prompt = f"""
你是一個資訊檢索與分析專家，請根據以下模型對比結果進行深入分析。
請特別關注 Top-1 accuracy 與 Recall@K 是否存在準確 vs. 覆蓋的平衡問題。

你需要根據以下面向來產出結果分析與討論完整報告：

1. 整體準確性比較（哪個模型 consistently 領先？）
2. 錯誤案例可能原因（哪些情況下兩者差異大）
3. 性能與成本比較（OpenAI API 的延遲和費用 vs. 本地部署BGE模型的計算開銷）
4. 結論： 綜合量化指標和質化分析，給出應用建議。


請以清晰條列方式產出分析報告。以下是指標結果：
{text}
"""
    response = client.chat.completions.create(
        model=os.getenv("AOAI_CHAT_DEPLOYMENT"),
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.4,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()





# ----- 主程式 -----

from collections import defaultdict

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute('SELECT id, question, answer, question_embedding_bge, question_embedding_openai, answer_embedding_bge, answer_embedding_openai, source_table FROM "2500567RAG" ORDER BY id;')
    rows = cur.fetchall()
    conn.close()

    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_table']].append(row)

    run_count = 1
    all_results = {}
    
    print(f"🔍 開始依照 source_table 共 {len(grouped)} 組資料進行評估...\n")
    for source, items in tqdm(grouped.items(), desc="評估中", unit="組"):
        bge_q = np.array([row["question_embedding_bge"] for row in items], dtype='float32')
        bge_a = np.array([row["answer_embedding_bge"] for row in items], dtype='float32')
        openai_q = np.array([row["question_embedding_openai"] for row in items], dtype='float32')
        openai_a = np.array([row["answer_embedding_openai"] for row in items], dtype='float32')
        ground_truth = list(range(len(items)))

        result_bge = evaluate_retrieval(build_faiss_index(bge_a), bge_q, ground_truth)
        result_openai = evaluate_retrieval(build_faiss_index(openai_a), openai_q, ground_truth)

        all_results[source] = {
            "BGE": {k: [v] for k, v in result_bge.items()},      
            "OpenAI": {k: [v] for k, v in result_openai.items()}
        }

    output_file = save_evaluation_to_txt(all_results, run_count)
    prompt_save_as(output_file)
    
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()
        print("\n[🧠] LLM 分析報告生成中...\n")
        summary = analyze_results_with_llm(content)
        print(summary)

        with open("LLM_分析報告.txt", "w", encoding="utf-8") as out:
            out.write(summary)


if __name__ == "__main__":
    main()
