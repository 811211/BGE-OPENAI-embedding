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
def fetch_documents(limit=100) -> List[Dict]:
    max_chars = 2000
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('SELECT document_id, text FROM "2500567RAG2" LIMIT %s;', (limit,))
    docs = []
    for row in cur.fetchall():
        doc_id = row[0]
        text = row[1]
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()[:max_chars]
        docs.append({"document_id": doc_id, "text": text})
    conn.close()
    return docs

# ----- 生成假問題 -----
def generate_questions_for_docs(docs: List[Dict], total_questions=100) -> List[Dict]:
    chat_deployment = os.getenv("AOAI_CHAT_DEPLOYMENT")
    questions = []
    num_docs = len(docs)
    if num_docs == 0:
        return []

    questions_per_doc = total_questions // num_docs
    extra = total_questions % num_docs

    for idx, doc in enumerate(tqdm(docs, desc="生成假問題中")):
        n = questions_per_doc + (1 if idx < extra else 0)
        for _ in range(n):
            prompt = (
                f"請針對以下內容，提出一個可以直接從內文找到答案的測試問題。\n"
                f"要求：\n"
                f"1️⃣ 問題僅根據內文，不做額外推測。\n"
                f"2️⃣ 問題簡短具體。\n"
                f"3️⃣ 使用繁體中文。\n"
                f"僅輸出問題內容，不要加「問題：」或其他說明。\n\n"
                f"內容：\n{doc['text']}"
            )
            try:
                response = client.chat.completions.create(
                    model=chat_deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=100
                )
                q = response.choices[0].message.content.strip()
                questions.append({
                    "document_id": doc["document_id"],
                    "question": q
                })
            except Exception as e:
                print(f"⚠️ 生成假問題失敗 document_id={doc['document_id']}，原因: {e}")
    return questions

def generate_answer(text: str, question: str) -> str:
    chat_deployment = os.getenv("AOAI_CHAT_DEPLOYMENT")
    prompt = (
        f"請根據以下內文，回答問題。\n"
        f"要求：\n"
        f"1️⃣ 答案必須簡短具體。\n"
        f"2️⃣ 僅使用繁體中文。\n"
        f"3️⃣ 答案內容必須直接來源於內文。\n"
        f"僅輸出答案本身，不要加上「答案：」或其他說明。\n\n"
        f"內文：\n{text}\n\n"
        f"問題：\n{question}"
    )
    try:
        response = client.chat.completions.create(
            model=chat_deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"⚠️ 生成答案失敗，原因: {e}")
        return ""




def save_fake_question(question: str, answer: str, document_id):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO "2500567RAG" (question, answer, document_id) VALUES (%s, %s, %s);',
        (question, answer, document_id)
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
def main():
    
    proceed = check_and_clear_table_if_needed()
    if not proceed:
        return
    
    print("抽取資料...")
    docs = fetch_documents(limit=50)
    print(f"共抽取 {len(docs)} 條資料")

    print("開始生成假問題...")
    questions = generate_questions_for_docs(docs, total_questions=100)
    print(f"共生成 {len(questions)} 條假問題")

    print("開始生成答案並寫入 2500567RAG 資料表...")
    for q in tqdm(questions, desc="保存假問題"):
        # 找到對應原始 text 來生成答案
        doc = next((d for d in docs if d["document_id"] == q["document_id"]), None)
        if doc:
            answer = generate_answer(doc["text"], q["question"])
            save_fake_question(
                question=q["question"],
                answer=answer,
                document_id=doc["document_id"]
            )
    print("✅ 假問題及答案已成功寫入資料庫！")


if __name__ == "__main__":
    main()
