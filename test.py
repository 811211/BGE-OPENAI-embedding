import psycopg2
import pandas as pd
import json
import re
from typing import List, Dict, Union  # Added Union for type hinting flexibility
import requests
import os
import threading
from queue import Queue, Empty
from io import StringIO
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from typing import List
import torch
from FlagEmbedding import BGEM3FlagModel  # Correct import for BGEM3FlagModel

load_dotenv()
# --- 資料庫連線配置 ---
# !!! 請替換為你們公司的實際 PSQL 連線資訊 !!!
DB_CONFIG = {
    "host": "20.210.159.117",
    "database": "postgres",
    "user": "readonlyuser",
    "password": "systex.6214",
    "port": "65432"  
}


def fetch_data_from_psql() -> List[Dict]:
    """
    從 PSQL 資料庫的指定 Table 提取數據並進行初步處理。
    將 headline 和 story 結合成 full_text。
    """
    conn = None
    all_processed_docs = []

    try:
        print("嘗試連接 PostgreSQL 資料庫...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("成功連接到 PostgreSQL 資料庫。")

        # --- 1. 處理新聞類的 Table: ntn001rtnews, ntn002rtnews, ntnx1netrtnews ---
        news_tables = ["ntn001rtnews", "ntn002rtnews", "ntnx1netrtnews"]
        # !!! 重要：請替換為你們新聞 Table 中實際的 ID 欄位名稱和日期/時間欄位名稱 !!!
        # 假設新聞 Table 有一個唯一的 ID 欄位叫 'article_id'
        # 假設有一個發布時間欄位叫 'publish_time'
        NEWS_ID_COLUMN = "id"  # 請替換

        for table_name in news_tables:
            print(f"\n正在從 Table '{table_name}' 提取數據...")
            # 這裡我用 LIMIT 1000 限制數據量，初期測試可以這樣做。
            # 正式抓取時，可能需要調整為沒有 LIMIT 或根據時間範圍來抓取。
            query = f"""
            SELECT "{NEWS_ID_COLUMN}", headline, story
            FROM {table_name}
            LIMIT 100;
            """

            try:
                cursor.execute(query)
                rows = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]  # 獲取欄位名稱

                for row in rows:
                    row_dict = dict(zip(column_names, row))

                    doc_id_raw = str(row_dict[NEWS_ID_COLUMN])
                    headline = (
                        row_dict.get("headline", "")
                        if row_dict.get("headline") is not None
                        else ""
                    )
                    story = (
                        row_dict.get("story", "")
                        if row_dict.get("story") is not None
                        else ""
                    )

                    # 結合 headline 和 story
                    # 如果 headline 和 story 都有，則用換行符隔開；否則只取非空的那個
                    full_text = (
                        f"{headline}\n\n{story}"
                        if headline and story
                        else (headline if headline else story)
                    )

                    if full_text.strip():  # 確保文本非空
                        all_processed_docs.append(
                            {
                                "original_id": f"{table_name}_{doc_id_raw}",  # 創建一個全局唯一的ID
                                "raw_text": full_text,
                                "source_table": table_name,
                                "type": "news",
                                "title": headline,  # 將 headline 也作為 title 元數據
                                "publish_time": str(
                                    row_dict.get("datetime", "")
                                ),  # 將時間轉換為字串
                            }
                        )
                print(f"成功從 '{table_name}' 提取 {len(rows)} 條記錄。")
            except psycopg2.Error as e_inner:
                print(f"從 Table '{table_name}' 提取數據時發生錯誤: {e_inner}")
                # 可以選擇跳過當前 Table 繼續處理下一個

        # --- 2. 處理研究報告 Table: rshcontent ---
        print("\n正在從 Table 'rshcontent' 提取數據...")
        # !!! 重要：請替換為你們研究報告 Table 中實際的 ID 欄位名稱 !!!
        RSH_ID_COLUMN = "reportid"  # 請替換

        query_rsh = f"""
        SELECT "{RSH_ID_COLUMN}", content
        FROM rshcontent
        LIMIT 100;
        """

        try:
            cursor.execute(query_rsh)
            rows_rsh = cursor.fetchall()
            column_names_rsh = [desc[0] for desc in cursor.description]

            for row_rsh in rows_rsh:
                row_dict_rsh = dict(zip(column_names_rsh, row_rsh))

                doc_id_raw_rsh = str(row_dict_rsh[RSH_ID_COLUMN])
                content = (
                    row_dict_rsh.get("content", "")
                    if row_dict_rsh.get("content") is not None
                    else ""
                )

                if content.strip():
                    all_processed_docs.append(
                        {
                            "original_id": f"rshcontent_{doc_id_raw_rsh}",
                            "raw_text": content,
                            "source_table": "rshcontent",
                            "type": "research_report",
                        }
                    )
            print(f"成功從 'rshcontent' 提取 {len(rows_rsh)} 條記錄。")
        except psycopg2.Error as e_inner:
            print(f"從 Table 'rshcontent' 提取數據時發生錯誤: {e_inner}")

    except psycopg2.Error as ex:
        print(f"資料庫連接或操作發生錯誤: {ex}")
    finally:
        if conn:
            conn.close()
            print("\nPostgreSQL 連接已關閉。")

    return all_processed_docs


def query_aoai_embedding(content: str) -> list[float]:
    """從 Azure OpenAI 服務獲取文本的 embedding 向量

    Args:
        content (str): 要進行 embedding 的文本內容

    Returns:
        list[float]: 返回 embedding 向量，如果發生錯誤則返回空列表
    """
    try_cnt = 2
    while try_cnt > 0:
        try_cnt -= 1
        api_key = os.getenv("EMBEDDING_API_KEY")
        api_base = os.getenv("EMBEDDING_URL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        print("load successfully embedding model:", embedding_model)
        try:
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_base,
            )
            embedding = client.embeddings.create(
                input=content,
                model=embedding_model,
            )
            return embedding.data[0].embedding
        except Exception as e:
            print(f"get_embedding_resource error | err_msg={e}")
    return []


import os
import threading
from typing import List, Dict, Union
import torch
from FlagEmbedding import BGEM3FlagModel  # Make sure this is imported

# Global model instance for BGE to avoid re-loading
bge_m3_model_instance = None
bge_model_lock = (
    threading.Lock()
)  # To ensure thread-safe loading if used in multi-thread


def load_bge_m3_model():
    """Loads the BGEM3FlagModel once."""
    global bge_m3_model_instance
    with bge_model_lock:
        if bge_m3_model_instance is None:
            try:
                print(
                    "Loading BGE-M3 model (FlagEmbedding)... This might take a moment if not cached."
                )
                # use_fp16=True can save memory and speed up on compatible GPUs,
                # but might cause issues on some CPU setups or older GPUs.
                bge_m3_model_instance = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
                print("BGE-M3 model loaded successfully.")
            except Exception as e:
                print(f"Error loading BGE-M3 model: {e}")
                bge_m3_model_instance = None
        return bge_m3_model_instance


# Corrected function signature for query_bge_embedding
def query_bge_embedding(
    content: Union[str, List[str]],
) -> Union[List[float], List[List[float]]]:
    """
    從本地載入的 BGE-M3 模型獲取文本的 embedding 向量。
    支援單個字串或字串列表輸入。

    Args:
        content (str | List[str]): 要進行 embedding 的文本內容或文本列表。

    Returns:
        List[float] (單個文本) 或 List[List[float]] (文本列表): 返回 embedding 向量。
    """
    model = load_bge_m3_model()
    if model is None:
        print("BGE-M3 model is not loaded. Cannot generate embedding.")
        return []

    try:
        # Based on BAAI example, access 'dense_vecs' key
        embeddings_output = model.encode(content)  # This returns a dictionary

        # Access the dense vectors from the dictionary
        dense_embeddings = embeddings_output["dense_vecs"]

        if isinstance(content, str):
            # If input was a single string, 'dense_vecs' will contain a single embedding array.
            # dense_embeddings will be a NumPy array of shape (embedding_dim,) if only one sentence.
            # If batch_size=1, it might still return (1, embedding_dim), so we need to handle that.
            # The BAAI example implies it will be a 2D array even for single input if batch_size is > 1
            # but if you pass a single string, it's typically (embedding_dim,).
            # To be safe, let's assume it always returns a 2D array like (num_sentences, embedding_dim)
            # and take the first (and only) embedding.
            if dense_embeddings.ndim == 2 and dense_embeddings.shape[0] == 1:
                return dense_embeddings[0].tolist()
            else:  # If it somehow returns 1D for single string
                return dense_embeddings.tolist()
        else:
            # If input was a list of strings, dense_embeddings will be a NumPy array of arrays
            return dense_embeddings.tolist()

    except Exception as e:
        print(f"query_bge_embedding error | err_msg={e}")
        return []


# --- 整合後續的清洗、分塊、保存到 JSONL 檔案的流程 ---
# （這部分與之前提供的程式碼相同，只是現在 raw_extracted_data 會來自 fetch_data_from_psql()）

# 1. 執行數據提取
raw_extracted_data = fetch_data_from_psql()
print(f"\n總共從資料庫提取到 {len(raw_extracted_data)} 條原始記錄。")

print(query_bge_embedding("這是一個測試文本。"))
print(query_aoai_embedding("這是一個測試文本。"))

# 2. 清洗與分塊 (假設 process_single_document 函數已定義)
# processed_docs_for_chunking = []
# for doc_item in raw_extracted_data:
#     processed_chunks = process_single_document(doc_item)
#     processed_docs_for_chunking.extend(processed_chunks)

# 3. 保存到 JSONL 檔案 (假設 save_to_jsonl 函數已定義)
# output_cleaned_chunks_file = 'company_financial_corpus_chunks.jsonl'
# with open(output_cleaned_chunks_file, 'w', encoding='utf-8') as f:
#     for item in processed_docs_for_chunking:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')
# print(f"\n所有清洗分塊後的數據已保存到 '{output_cleaned_chunks_file}'，共 {len(processed_docs_for_chunking)} 個塊。")
