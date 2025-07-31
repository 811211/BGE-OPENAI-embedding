# ===========================================
# 📄 demo5.py
# 功能：
# 1. 從 PostgreSQL 資料庫讀取問題與答案及其 embedding
# 2. 使用 FAISS 建立索引並進行模型檢索評估
# 3. 評估指標包含 Precision@K, Top-K Accuracy, Mean Rank, MRR
# 4. 分析錯誤案例並儲存詳細錯誤資訊（含原始問題、正確答案、檢索結果等）
# 5. 呼叫 LLM 分析模型表現，並輸出完整報告與統計資訊
# ===========================================

import os
import shutil
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
from typing import List, Dict

import numpy as np
import faiss
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from openai import AzureOpenAI
from FlagEmbedding import BGEM3FlagModel

# TODO: 以下區塊建議整理為功能區塊，例如
# - 環境配置與初始化
# - 相關工具函數
# - embedding 函數
# - 建立 FAISS 索引
# - 評估指標計算
# - 資料儲存
# - LLM 分析與報告生成
# - 主程式

# ----- 環境配置 -----

load_dotenv()

print("DEBUG: AOAI_KEY =", os.getenv("AOAI_KEY"))
print("DEBUG: AOAI_ENDPOINT =", os.getenv("AOAI_ENDPOINT"))
print("DEBUG: AOAI_API_VERSION =", os.getenv("AOAI_API_VERSION"))
print("DEBUG: AOAI_CHAT_DEPLOYMENT =", os.getenv("AOAI_CHAT_DEPLOYMENT"))

DB_CONFIG = {
    "host": "210.67.12.103",
    "database": "Learning",
    "user": "editor",
    "password": "systex.6214",
    "port": 65432
}

embedding_stats = {
    "OpenAI": {"total_time": 0.0, "count": 0, "token_usage": 0},
    "BGE": {"total_time": 0.0, "count": 0}
}


# -----初始化 AzureOpenAI client-----

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

BGE_MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)



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

def embed_openai(raw_text: str) -> np.ndarray:
    try:
        start = time.time()
        response = embedding_ada.embeddings.create(
            input=raw_text,
            model=os.getenv("EMBEDDING_MODEL")
        )
        duration = time.time() - start
        print(f"[⏱️ OpenAI] 單筆處理時間: {duration:.4f} 秒")
        
        # ✅ 累加統計
        embedding_stats["OpenAI"]["total_time"] += duration
        embedding_stats["OpenAI"]["count"] += 1
        embedding_stats["OpenAI"]["token_usage"] += response.usage.total_tokens
        
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        print(f"embed_openai 出錯: {e}")
        return np.zeros((1536,), dtype="float32")

def embed_bge(raw_text: str) -> np.ndarray:
    try:
        start = time.time()
        output = BGE_MODEL.encode(raw_text)
        duration = time.time() - start
        print(f"[⏱️ BGE] 單筆處理時間: {duration:.4f} 秒")
        
        # ✅ 累加統計
        embedding_stats["BGE"]["total_time"] += duration
        embedding_stats["BGE"]["count"] += 1
        
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
    faiss.normalize_L2(vectors)    # COSINE 相似度需要先歸一化 要改L2直接註解
    index = faiss.IndexFlatIP(dim) 
    index.add(vectors)
    return index

# ----- 評估指標計算 -----

def save_error_cases(details: list, id2text: dict = None, output_dir: str = "report", filename: str = "error_cases.csv"):
    error_cases = [d for d in details if not d.get("hit", False)]
    
    rows = []
    for d in error_cases:
        row = {
            "query_id": d["query_id"],
            "ground_truth": d["ground_truth"],
            "retrieved_ids": d["retrieved_ids"],
            "similarities": d["similarities"],
            "rank": d["rank"],
            "precision": d["precision"],
            "mrr": d["reciprocal_rank"]
        }

        if id2text:
            meta = id2text.get(d["ground_truth"], {})
            row["ground_truth_text"] = meta.get("raw_text", "")
            row["query_text"] = id2text.get(d["query_id"], {}).get("query", "")  
            row["original_id"] = meta.get("original_id", "")
            row["source_table"] = meta.get("source_table", "")
            row["retrieved_texts"] = [
                id2text.get(rid, {}).get("raw_text", "") for rid in d["retrieved_ids"]
            ]


        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"❌ 錯誤案例已儲存：{output_path}")
    


def evaluate_retrieval(index, queries, ground_truth, k=5, id2text=None, index_to_doc_id=None):

    D, I = index.search(queries, k)

    precision_at_k = []
    top_k_accuracy = []
    top1_accuracy = []  
    mean_rank = []
    reciprocal_ranks = []
    details = []

    for i, retrieved in enumerate(I):
        relevant = ground_truth[i]
        retrieved_list = list(retrieved)
        retrieved_doc_ids = [index_to_doc_id[idx] for idx in retrieved_list]

        detail = {
            "query_id": i,
            "ground_truth": relevant,
            "retrieved_ids": retrieved_doc_ids,
            "similarities": [float(x) for x in D[i]],
            "hit": False,
            "rank": None,
            "precision": 0,
            "mean_rank": k + 1,
            "reciprocal_rank": 0,
            "top1_hit": False
        }

        try:
            rank = retrieved_doc_ids.index(relevant)
            precision_at_k.append(1 / k)
            top_k_accuracy.append(1)
            top1_accuracy.append(1 if rank == 0 else 0)
            mean_rank.append(rank + 1)
            reciprocal_ranks.append(1 / (rank + 1))
            detail.update({
                "hit": True,
                "rank": rank + 1,
                "precision": round(1 / k, 4),
                "mean_rank": rank + 1,
                "reciprocal_rank": round(1 / (rank + 1), 4),
                "top1_hit": rank == 0
            })
        except ValueError:
            precision_at_k.append(0)
            top_k_accuracy.append(0)
            top1_accuracy.append(0)
            mean_rank.append(k + 1)
            reciprocal_ranks.append(0)

        details.append(detail)
    save_error_cases(details, id2text=id2text)

    return {
        "Precision@K": round(np.mean(precision_at_k), 4),
        "Top-K Accuracy": round(np.mean(top_k_accuracy), 4),
        "Top-1 Accuracy": round(np.mean(top1_accuracy), 4),
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
    header = f"{'Query':<6}{'Rank':<6}{'Top1':<6}{'Precision':<10}{'MRR':<8}"
    lines.append(header)
    lines.append("-" * 90)

    for d in details:
        # 每筆主要資訊（query ID、排名、精確度等）
        line = f"{d['query_id']:<6}{str(d['rank'] or '-'):<6}{'✅' if d['top1_hit'] else '❌':<6}"
        line += f"{d['precision']:<10}{d['reciprocal_rank']:<8}"
        lines.append(line)

        # 顯示 ground-truth 與 TopK ID（單獨顯示，避免擠壓）
        lines.append(f"{'GT:':<6}{d['ground_truth']}")
        lines.append(f"{'TopK:':<6}{', '.join(d['retrieved_ids'])}")
        lines.append("-" * 90)

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
                
        # 嵌入計算時間與成本統計
        f.write("[⏱️] 嵌入計算時間與成本統計\n")
        f.write("=" * 80 + "\n")
        for model, stat in embedding_stats.items():
            f.write(f"🔹 {model} 模型\n")
            f.write(f"總處理時間：{round(stat['total_time'], 2)} 秒\n")
            f.write(f"總筆數：{stat['count']} 筆\n")
            if stat["count"] > 0:
                avg_time = round(stat['total_time'] / stat['count'], 4)
                f.write(f"平均時間：{avg_time} 秒/筆\n")
            if model == "OpenAI":
                tokens = stat.get("token_usage", 0)
                cost = round(tokens / 1000 * 0.0001, 6)  # 自訂單價
                f.write(f"Token 使用量：{tokens} tokens\n")
                f.write(f"成本估算：${cost} USD\n")
            else:
                f.write("本地模型無 token 成本\n")
            f.write("\n")

        # LLM 分析區段
        f.write("[🧠] LLM 模型分析報告\n")
        f.write("=" * 80 + "\n")
        f.write(summary_text + "\n\n")

        # Retrieval 詳細記錄
        f.write("[🔍] Retrieval 詳細過程記錄\n")
        f.write("=" * 80 + "\n")
        for source, models in results_dict.items():
            for model_name, metric_data in models.items():
                if "Details" in metric_data:
                    f.write(f"📂 {source} - {model_name} Retrieval Details:\n")
                    raw_text = format_details_human_readable(metric_data["Details"])
                    f.write(raw_text + "\n\n")               

    return filename


# ----- LLM 分析 -----

def analyze_results_with_llm(results_dict: dict) -> str:
    import json
    raw_text = json.dumps(results_dict, ensure_ascii=False, indent=2)

    analysis_prompt = f"""
你是一個資訊檢索與分析專家，請根據以下模型對比結果進行深入分析。

你需要根據以下面向並嚴格依照模型對比結果來產出結果分析與討論完整報告：

1. 整體準確性比較（哪個模型 consistently 領先？）
2. 錯誤案例可能原因（哪些情況下兩者差異大）
3. 性能與成本比較（OpenAI API 的延遲和費用 vs. 本地部署BGE模型的計算開銷）
4. 結論： 綜合量化指標和質化分析，給出應用建議。



請以清晰條列方式產出分析報告。以下是指標結果：
{raw_text}
"""
    response = client.chat.completions.create(
        model=os.getenv("AOAI_CHAT_DEPLOYMENT"),
        messages=[{"role": "user", "content": analysis_prompt}],
        temperature=0.4,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


# ----- 主程式 -----



def main():
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute('SELECT id, original_id, question, source_table, raw_text FROM "2500567RAG" ORDER BY id;')
    rows = cur.fetchall()
    conn.close()

    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_table']].append(row)

    run_count = 1  # 控制次數
    all_results = {}
    
    print(f"🔍 開始依照 source_table 共 {len(grouped)} 組資料進行評估...\n")
    for source, items in tqdm(grouped.items(), desc="評估中", unit="組"):
        
        # 即時計算向量
        bge_q = np.array([embed_bge(row["question"]) for row in items], dtype='float32')
        bge_a = np.array([embed_bge(row["raw_text"]) for row in items], dtype='float32')
        openai_q = np.array([embed_openai(row["question"]) for row in items], dtype='float32')
        openai_a = np.array([embed_openai(row["raw_text"]) for row in items], dtype='float32')
        doc_ids = [row["original_id"] for row in items]
        ground_truth = doc_ids

        # 建立 FAISS index 時的順序 → doc_id 對照
        index_to_doc_id = {i: doc_id for i, doc_id in enumerate(doc_ids)}
                
        id2text = {
            row["original_id"]: {
                "raw_text": row["raw_text"],                   # 整篇原文
                "query": row["question"],              # 問題
                "original_id": row["original_id"],
                "source_table": row["source_table"]
            }
            for row in items
        }
 

        result_bge = evaluate_retrieval(build_faiss_index(bge_a), bge_q, ground_truth, id2text=id2text, index_to_doc_id=index_to_doc_id)
        result_openai = evaluate_retrieval(build_faiss_index(openai_a), openai_q, ground_truth, id2text=id2text, index_to_doc_id=index_to_doc_id)

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
