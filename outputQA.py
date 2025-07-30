# ===========================================
# 📄 outputQA
# 功能：生產問題和答案，並儲存到資料庫
# ===========================================
import os
import psycopg2
import json
from typing import List, Dict
from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

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
def generate_questions_for_docs(docs: List[Dict], total_questions=1000) -> List[Dict]:
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

                    if q in existing_questions:
                        retry += 1
                        continue  # 重複問題就重試
                    else:
                        existing_questions.add(q)
                        questions.append({
                            "document_id": doc["document_id"],
                            "question": q
                        })
                        break  # 成功生成問題就跳出 retry 迴圈

                except Exception as e:
                    print(f"⚠️ 生成假問題失敗 document_id={doc['document_id']}，原因: {e}")
                    break  # 若是 LLM API 失敗就跳出 retry
    return questions

def save_question(question: str, document_id: str, source_table: str, text: str):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO "2500567RAG" (question, document_id, source_table, text) VALUES (%s, %s, %s, %s);',
        (question, document_id, source_table, text)
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
            resp = input("清除現有資料並重新處理？(y/N): ")
            if resp.lower() == 'y':
                cur.execute('DELETE FROM public."2500567RAG"')
                cur.execute('TRUNCATE TABLE public."2500567RAG" RESTART IDENTITY;')
                conn.commit()
                print("✅ 已清除資料")
            else:
                print("取消處理")
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
# 更新後主程式
def main():
    proceed = check_and_clear_table_if_needed()
    if not proceed:
        return

    print("開始依照 source_table 類型抽取資料...")
    grouped_docs = fetch_documents(limit=1000)

    total_all_questions = 0
    for source_table, docs in grouped_docs.items():
        print(f"\n🗂️ 類型: {source_table}，共 {len(docs)} 筆")
        print("🔄 開始生成問題...")
        questions = generate_questions_for_docs(docs, total_questions=1000)
        print(f"✅ 共為類型 {source_table} 生成 {len(questions)} 筆問題")

        for q in tqdm(questions, desc=f"{source_table} - 寫入中"):
            doc = next((d for d in docs if d["document_id"] == q["document_id"]), None)
            if doc:
                save_question(
                    question=q["question"],
                    document_id=doc["document_id"],
                    source_table=doc["source_table"],
                    text=doc["text"]
                )

        total_all_questions += len(questions)

    print(f"\n✅ 全部類型問題生成完畢！總數: {total_all_questions} 筆")



if __name__ == "__main__":
    main()
