import os
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from utils.database_config import get_database_config
from utils.ai_client import get_embedding_for_content

DB_CONFIG = {
    "host": "210.67.12.103",
    "database": "Learning",
    "user": "editor",
    "password": "systex.6214",
    "port": "65432"
}

class WenWangQianProcessor:
    """文王籤處理器"""

    def __init__(self):
        self.db_config = {
            "host": "210.67.12.103",
            "database": "Learning",
            "user": "editor",
            "password": "systex.6214",
            "port": 65432  
        }
        self.max_workers = 4
        self.lock = threading.Lock()

    def read_text(self, file_path: str) -> List[Dict]:
        """讀取文王籤.txt 並按籤號分割"""
        print(f"📖 正在讀取文王籤資料: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # 使用正則分段
            pattern = r'(\d+：.*?)(?=\n\d+：|\Z)'  # 捕捉一籤到下一籤之間的文字
            matches = re.findall(pattern, text, re.DOTALL)
            results = []
            for match in matches:
                m = re.match(r'(\d+)：(.*)', match, re.DOTALL)
                if m:
                    num = m.group(1)
                    content = m.group(2).strip().replace('\n', ' ')
                    context = f"文王籤 籤號 {num} "
                    results.append({
                        'content': content,
                        'context': context,
                        'chunk_index': int(num)
                    })
            print(f"✅ 讀取完成，共 {len(results)} 首籤詩")
            return results
        except Exception as e:
            print(f"❌ 讀取文王籤.txt 失敗: {e}")
            return []

    def query_aoai_embedding(self, content: str) -> list[float]:
        return get_embedding_for_content(content)

    def process_embedding_batch(self, chunks_batch: List[Dict], batch_id: int, total_batches: int) -> List[Dict]:
        processed_chunks = []
        for i, chunk in enumerate(chunks_batch):
            try:
                embedding = self.query_aoai_embedding(chunk['content'])
                chunk['embedding'] = embedding
                with self.lock:
                    print(f"🧵 執行緒 {batch_id}/{total_batches} | 籤 {chunk['chunk_index']}")
                processed_chunks.append(chunk)
                time.sleep(0.1)
            except Exception as e:
                with self.lock:
                    print(f"❌ 執行緒 {batch_id} 發生錯誤: {e}")
                chunk['embedding'] = None
                processed_chunks.append(chunk)
        return processed_chunks

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        print(f"🚀 開始生成 embedding，共 {len(chunks)} 首籤詩")
        batch_size = len(chunks) // self.max_workers + 1
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        all_processed_chunks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_embedding_batch, b, i+1, len(batches)): i for i, b in enumerate(batches)}
            for future in as_completed(futures):
                all_processed_chunks.extend(future.result())

        all_processed_chunks.sort(key=lambda x: x['chunk_index'])
        print(f"✅ embedding 生成完成，共 {len(all_processed_chunks)} 條")
        return all_processed_chunks

    def create_database_tables(self):
        print("正在創建資料庫表格...")
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.\"2500567RAG\"
                (
                    id SERIAL NOT NULL,
                    embedding_vector double precision[],
                    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
                    content text COLLATE pg_catalog."default",
                    context text NOT NULL,
                    CONSTRAINT embeddings_pkey PRIMARY KEY (id)
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_content ON \"2500567RAG\" USING gin(to_tsvector('chinese', content));
                CREATE INDEX IF NOT EXISTS idx_embeddings_context ON \"2500567RAG\"(context);
            """)
            conn.commit()
            cur.close()
            conn.close()
            print("✅ 資料庫表格就緒")
        except Exception as e:
            print(f"❌ 創建資料庫時錯誤: {e}")

    def save_to_database(self, chunks: List[Dict]):
        print("💾 儲存到資料庫...")
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            embedding_data = [(c['embedding'], c['content'], c['context']) for c in chunks if c.get('embedding')]
            execute_values(cur, """
                INSERT INTO public.\"2500567RAG\" (embedding_vector, content, context) VALUES %s
            """, embedding_data)
            conn.commit()
            cur.close()
            conn.close()
            print(f"✅ 儲存完成，共 {len(embedding_data)} 筆記錄")
        except Exception as e:
            print(f"❌ 儲存資料庫時錯誤: {e}")

    def check_existing_data(self) -> int:
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM public.\"2500567RAG\"")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            return count
        except Exception as e:
            print(f"檢查資料庫錯誤: {e}")
            return 0

    def process_text(self, file_path: str):
        print("=" * 50)
        print("開始處理文王籤資料")
        print("=" * 50)

        self.create_database_tables()
        existing_count = self.check_existing_data()
        if existing_count > 0:
            print(f"⚠️ 資料庫已有 {existing_count} 筆記錄")
            resp = input("清除現有資料並重新處理？(y/N): ")
            if resp.lower() == 'y':
                try:
                    conn = psycopg2.connect(**self.db_config)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM public.\"2500567RAG\"")
                    conn.commit()
                    cur.close()
                    conn.close()
                    print("✅ 已清除資料")
                except Exception as e:
                    print(f"清除資料失敗: {e}")
                    return
            else:
                print("取消處理")
                return

        chunks = self.read_text(file_path)
        if not chunks:
            print("❌ 讀取失敗，中止執行")
            return

        chunks_emb = self.generate_embeddings(chunks)
        self.save_to_database(chunks_emb)
        print("=" * 50)
        print("📦 文王籤處理完成")
        print("=" * 50)
        final_count = self.check_existing_data()
        print(f"📊 現在資料庫中有 {final_count} 筆記錄")

def main():
    processor = WenWangQianProcessor()
    file_path = "文王籤.txt"
    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案: {file_path}")
        return
    processor.process_text(file_path)

if __name__ == "__main__":
    main()