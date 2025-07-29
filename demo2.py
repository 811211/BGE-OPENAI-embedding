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
    
def wrap_metrics(metrics: dict) -> dict:
    result = {}
    for k, v in metrics.items():
        if k == "Details":
            result[k] = v  # 不包成 list
        else:
            result[k] = [v]  # 正常指標包成 list
    return result

    

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
    details = []

    for i, retrieved in enumerate(I):
        relevant = ground_truth[i]
        retrieved_list = list(retrieved)
        detail = {
            "query_id": i,
            "ground_truth": relevant,
            "retrieved_ids": [int(x) for x in retrieved_list],
            "similarities": [float(x) for x in D[i]],
            "hit": False,
            "rank": None,
            "recall": 0,
            "precision": 0,
            "mean_rank": k + 1,
            "reciprocal_rank": 0,
            "top1_hit": False
        }
    
        try:
            rank = retrieved_list.index(relevant)
            recall_at_k.append(1)
            precision_at_k.append(1 / (rank + 1))
            # top_k_accuracy.append(1 if rank == 0 else 0) # Top-1 Accuracy rank == 0 → Top-1 命中。
            top_k_accuracy.append(1 if rank < k else 0)    # Top-K Accuracy rank <  k → Top-k 命中
            mean_rank.append(rank + 1)
            reciprocal_ranks.append(1 / (rank + 1))
            detail.update({
                "hit": True,
                "rank": rank + 1,
                "recall": 1,
                "precision": round(1 / (rank + 1), 4),
                "mean_rank": rank + 1,
                "reciprocal_rank": round(1 / (rank + 1), 4),
                "top1_hit": rank == 0
            })
        except ValueError:
            recall_at_k.append(0)
            precision_at_k.append(0)
            top_k_accuracy.append(0)
            mean_rank.append(k + 1)
            reciprocal_ranks.append(0)
            
        details.append(detail)

    return {
        "Recall@K": round(np.mean(recall_at_k), 4),
        "Precision@K": round(np.mean(precision_at_k), 4),
        "Top-K Accuracy": round(np.mean(top_k_accuracy), 4),
        "Mean Rank": round(np.mean(mean_rank), 4),
        "MRR": round(np.mean(reciprocal_ranks), 4),
        "Details": details
    }

# ----- 資料儲存 -----

# def save_evaluation_to_txt(results: dict, run_count: int):
#     filename = f"BGE-OPENAI-embedding test({run_count}).txt"
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(f"BGE-OPENAI Embedding Evaluation Report (Runs = {run_count})\n")
#         f.write("=" * 60 + "\n\n")
#         for source, models in results.items():
#             f.write(f"📄 Source Table: {source}\n")
#             f.write("-" * 60 + "\n")
#             for model_name, metric_data in models.items():
#                 f.write(f"🔹 {model_name}:\n")
#                 for metric, values in metric_data.items():
#                     if run_count == 1:
#                         f.write(f"{metric}: {values[0]}\n")
#                     else:
#                         avg = round(sum(values) / len(values), 4)
#                         f.write(f"{metric}: Run1={values[0]}  Run2={values[1]}  Run3={values[2]}  Avg={avg}\n")
#                 f.write("\n")
#     return filename


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
        
def format_details_human_readable(details: List[Dict]) -> str:
    lines = []
    header = f"{'Query':<6}{'GT':<6}{'TopK IDs':<30}{'Rank':<6}{'Top1':<6}{'Recall':<8}{'Prec':<8}{'MRR':<8}"
    lines.append(header)
    lines.append("-" * len(header))
    for d in details:
        line = f"{d['query_id']:<6}{d['ground_truth']:<6}{str(d['retrieved_ids'])[:28]:<30}{str(d['rank'] or '-'):<6}"
        line += f"{'✅' if d['top1_hit'] else '❌':<6}{d['recall']:<8}{d['precision']:<8}{d['reciprocal_rank']:<8}"
        lines.append(line)
    return "\n".join(lines)

        
def save_full_report(results_dict, summary_text, run_count):
    import json

    filename = f"完整評估報告_Run{run_count}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        # 標題
        f.write(f"[📊] BGE-OPENAI Embedding Evaluation Report (Run {run_count})\n")
        f.write("=" * 80 + "\n\n")

        # 每組結果
        for source, models in results_dict.items():
            f.write(f"📄 Source Table: {source}\n")
            f.write("-" * 80 + "\n")
            for model_name, metric_data in models.items():
                f.write(f"🔹 {model_name} 模型:\n")
                for metric, values in metric_data.items():
                    if run_count == 1:
                        f.write(f"{metric}: {values[0]}\n")
                    else:
                        avg = round(sum(values) / len(values), 4)
                        f.write(f"{metric}: Run1={values[0]}  Run2={values[1]}  Run3={values[2]}  Avg={avg}\n")
                f.write("\n")

        f.write("\n[🧠] LLM 模型分析報告\n")
        f.write("=" * 80 + "\n")
        f.write(summary_text + "\n\n")

        f.write("[🔍] Retrieval 詳細過程記錄\n")
        f.write("=" * 80 + "\n")

        for source, models in results_dict.items():
            for model_name, metric_data in models.items():
                if "Details" in metric_data:
                    f.write(f"📂 {source} - {model_name} Retrieval Details:\n")
                    text = format_details_human_readable(metric_data["Details"])
                    f.write(text + "\n\n")               
    return filename

# ----- LLM 分析 -----

def analyze_results_with_llm(results_dict: dict) -> str:
    import json
    text = json.dumps(results_dict, ensure_ascii=False, indent=2)

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
            "BGE": wrap_metrics(result_bge),
            "OpenAI": wrap_metrics(result_openai)
        }

    # Step 1: 產生分析報告
    print("\n[🧠] LLM 分析報告生成中...\n")

    # 過濾掉 Details，只保留指標數值
    summary_input = {
        source: {
            model: {k: v for k, v in metrics.items() if k != "Details"}
            for model, metrics in models.items()
        }
        for source, models in all_results.items()
    }

    # 丟給 LLM 分析
    summary = analyze_results_with_llm(summary_input)
    print(summary)


    # Step 2: 整合成單一完整報告
    full_report_path = save_full_report(all_results, summary, run_count)

    # Step 3: 提供使用者另存
    prompt_save_as(full_report_path)



if __name__ == "__main__":
    main()
