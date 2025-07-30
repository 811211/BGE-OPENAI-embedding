# ===========================================
# 📄 question_generator_pipeline.py
# 功能說明：
# - 從 PostgreSQL 中依據不同 source_table 抽取文件樣本。
# - 使用 Azure OpenAI 針對每段文件自動生成繁體中文問題。
# - 從原文中擷取唯一答案段落。
# - 將問題、答案與原文存回指定資料表。
# ===========================================

# ----- 📦 標準套件 -----
import os
import json
from typing import List, Dict

# ----- 📦 第三方套件 -----
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI

# ----- 環境配置 -----
load_dotenv()

print("DEBUG: AOAI_KEY =", os.getenv("AOAI_KEY"))
print("DEBUG: AOAI_ENDPOINT =", os.getenv("AOAI_ENDPOINT"))
print("DEBUG: AOAI_API_VERSION =", os.getenv("AOAI_API_VERSION"))
print("DEBUG: AOAI_CHAT_DEPLOYMENT =", os.getenv("AOAI_CHAT_DEPLOYMENT"))

client = AzureOpenAI(
    api_key=os.getenv("AOAI_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_version=os.getenv("AOAI_API_VERSION")
)

DB_CONFIG = {
    "host": "210.67.12.103",
    "database": "Learning",
    "user": "editor",
    "password": "systex.6214",
    "port": 65432
}

# ----- 抽取文件 -----
def fetch_documents(limit=1000) -> Dict[str, List[Dict]]:
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

# ----- 生成問題 -----
def generate_questions_for_docs(docs: List[Dict], total_questions=100) -> List[Dict]:
    chat_deployment = os.getenv("AOAI_CHAT_DEPLOYMENT")
    questions = []
    existing_questions = set()
    num_docs = len(docs)
    if num_docs == 0:
        return []

    questions_per_doc = total_questions // num_docs
    extra = total_questions % num_docs

    for idx, doc in enumerate(tqdm(docs, desc="生成假問題中")):
        n = questions_per_doc + (1 if idx < extra else 0)

        for _ in range(n):
            retry = 0
            max_retry = 3

            while retry < max_retry:
                prompt = (
                    f"請針對以下內容，提出一個可以直接從內文找到答案的測試問題，並包含『能唯一指涉答案的關鍵字』，"
                    f"例如數值、公司名、年份或具體事件等，使問題明確對應單一答案。\n"
                    f"要求：\n"
                    f"1️⃣ 問題僅根據內文，不做額外推測。\n"
                    f"2️⃣ 問題簡短具體。\n"
                    f"3️⃣ 使用繁體中文。\n"
                    f"4️⃣ 僅輸出問題內容，不要加「問題：」或其他說明。\n\n"
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
                        continue
                    else:
                        existing_questions.add(q)
                        questions.append({
                            "document_id": doc["document_id"],
                            "question": q,
                            "answer": a
                        })
                        break

                except Exception as e:
                    print(f"⚠️ 生成假問題失敗 document_id={doc['document_id']}，原因: {e}")
                    break
    return questions

# ----- 從原文擷取答案 -----
def extract_answer_from_text(text: str, question: str, client=None, deployment=None) -> str:
    prompt = (
        f"""根據以下內容與問題，請從內文中擷取**最精準的一段文字**作為答案，不要自行改寫或補充。\n
        答案必須是原文中的一段話。\n
        答案格式為：**「關鍵詞:數字」**，並以 JSON 格式輸出。\n\n
        {text}\n\n
        【問題】\n{question}\n\n
        【輸出】\n{{"answer": "股價變化:500"}}
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
        return ""

# ----- 寫入資料 -----
def save_question(entry: Dict):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO "2500567RAG" (document_id, source_table, question, answer, text) VALUES (%s, %s, %s, %s, %s);',
        (
            entry["document_id"],
            entry["source_table"],
            entry["question"],
            entry["answer"],
            entry["text"]
        )
    )
    conn.commit()
    conn.close()

# ----- 檢查資料表是否清除 -----
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
            
            if resp == 'y':
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
                print("⚠️ 無效選項，請重新執行")
                cur.close()
                conn.close()
                return False
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"檢查/清除資料失敗: {e}")
        return False

# ----- 主程式入口 -----
def main():
    if not check_and_clear_table_if_needed():
        return

    print("📥 開始抽取資料庫文件...")
    source_table_docs = fetch_documents(limit=1000)

    for source_table, docs in source_table_docs.items():
        print(f"📚 處理來源：{source_table}（共 {len(docs)} 筆）")
        generated = generate_questions_for_docs(docs, total_questions=100)

        for entry in tqdm(generated, desc=f"💾 儲存 {source_table} 問答對"):
            save_question({
                "document_id": entry["document_id"],
                "source_table": source_table,
                "question": entry["question"],
                "answer": entry["answer"],
                "text": next((d["text"] for d in docs if d["document_id"] == entry["document_id"]), "")
            })

    print("✅ 所有資料處理完成")

if __name__ == "__main__":
    main()
