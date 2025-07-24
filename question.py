import os
import psycopg2
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
from typing import List, Dict
from openai import AzureOpenAI
from dotenv import load_dotenv

# ----- 環境配置 -----
load_dotenv()

embedding_model = os.getenv("EMBEDDING_MODEL")

print("DEBUG: EMBEDDING_API_KEY =", os.getenv("EMBEDDING_API_KEY"))
print("DEBUG: EMBEDDING_URL =", os.getenv("EMBEDDING_URL"))
print("DEBUG: AOAI_API_VERSION =", os.getenv("AOAI_API_VERSION"))
print("DEBUG: EMBEDDING_MODEL =", os.getenv("EMBEDDING_MODEL"))
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

# ----- DB 連線參數 -----
DB_CONFIG = {
    "host": "210.67.12.103",
    "database": "Learning",
    "user": "editor",
    "password": "systex.6214",
    "port": 65432
}

# ----- 抽取資料庫文件 -----
def fetch_documents(limit=100) -> List[Dict]:
    max_chars = 2000  # 每條記錄的最大字符數
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('SELECT text FROM "2500567RAG2" LIMIT %s;', (limit,))
    docs = []
    for idx, row in enumerate(cur.fetchall()):
        text = row[0]
        # 安全檢查
        if not isinstance(text, str):
            print(f"⚠️ WARNING: 非 string 資料發現，row={row}")
            text = str(text)
        text = text.strip()[:max_chars]
        docs.append({"id": idx, "text": text})
    conn.close()
    return docs

    

# ----- 生成假問題 (Azure OpenAI 版本) -----
def generate_questions_for_docs(docs: List[Dict], num_questions=1):
    questions = []
    chat_deployment = os.getenv("AOAI_API_VERSION")
    for doc in docs:
        prompt = f"請根據以下內容，生成一個與內容有關的測試問題（以繁體中文描述）：\n\n{doc['text']}\n\n問題："
        try:
            response = client.chat.completions.create(
                model=chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=100
            )
            q = response.choices[0].message.content.strip()
            questions.append({"doc_id": doc["id"], "question": q})
        except Exception as e:
            print(f"生成問題出錯 (AOAI): {e}")
    return questions

# ----- Embedding 函數 -----
def embed_openai(text: str) -> np.ndarray:
    if not isinstance(text, str):
        print(f"embed_openai 警告: text 非 str，實際類型 {type(text)}，值={text}")
        text = str(text)
    try:
        response = embedding_ada.embeddings.create(
            input=text,
            model=embedding_model
        )
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        print(f"embed_openai 出錯: {e}")
        return np.zeros((1536,), dtype="float32")

def embed_bge(text: str) -> np.ndarray:
    if not isinstance(text, str):
        print(f"embed_bge 警告: text 非 str，實際類型 {type(text)}，值={text}")
        text = str(text)
    try:
        output = BGE_MODEL.encode(text)
        dense_vecs = output.get("dense_vecs")
        if dense_vecs is None:
            print(f"embed_bge 警告: dense_vecs 為空，text={text[:50]}...")
            return np.zeros((1024,), dtype="float32")
        # ✅ 這裡直接 return，不用再取 [0]
        return dense_vecs.astype("float32")
    except Exception as e:
        print(f"embed_bge 出錯: {e}")
        return np.zeros((1024,), dtype="float32")


# ----- 建立 FAISS index -----
def build_faiss_index(vectors: List[np.ndarray]):
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(vectors))
    return index

# ----- Pipeline 主程式 -----
def main():
    print("抽取資料中...")
    docs = fetch_documents(limit=50)
    print(f"共抽取 {len(docs)} 條資料")

    print("生成假問題 (AOAI)...")
    questions = generate_questions_for_docs(docs)

    print("計算 AOAI embedding...")
    vectors_openai = [embed_openai(doc["text"]) for doc in tqdm(docs)]

    print("計算 BGE embedding...")
    vectors_bge = [embed_bge(doc["text"]) for doc in tqdm(docs)]

    print("建立 FAISS index...")
    print(f"vectors_bge 長度 = {len(vectors_bge)}")
    for idx, vec in enumerate(vectors_bge):
        print(f"第 {idx} 個 vector shape = {vec.shape} dtype={vec.dtype}")

    index_openai = build_faiss_index(vectors_openai)
    index_bge = build_faiss_index(vectors_bge)

    print("開始評估...")
    k = 1
    correct_openai = 0
    correct_bge = 0

    for q in tqdm(questions):
        try:
            q_vec_openai = embed_openai(q["question"])
            q_vec_bge = embed_bge(q["question"])

            D_openai, I_openai = index_openai.search(q_vec_openai.reshape(1, -1), k)
            D_bge, I_bge = index_bge.search(q_vec_bge.reshape(1, -1), k)

            doc_idx = next(i for i, d in enumerate(docs) if d["id"] == q["doc_id"])
            
            if doc_idx in I_openai[0]:
                correct_openai += 1
            if doc_idx in I_bge[0]:
                correct_bge += 1
        except Exception as e:
            print(f"評估時發生錯誤: {e}")

    acc_openai = correct_openai / len(questions) if questions else 0
    acc_bge = correct_bge / len(questions) if questions else 0

    print(f"AOAI embedding 檢索正確率: {acc_openai:.2f}")
    print(f"BGE embedding 檢索正確率: {acc_bge:.2f}")

if __name__ == "__main__":
    
    # test_text = "<自營商買超30-11>碳高息(439),金 寶(387),矽 創(37) ..."
    # print(f"測試文本: {test_text[:50]}...")

    # vec = embed_openai(test_text)
    # print(f"Embedding 向量 shape: {vec.shape}")
    # print(f"Embedding 前 5 維: {vec[:5]}")
    
    # test_text = "<自營商買超30-11>碳高息(439),金 寶(387),矽 創(37)..."
    # print(f"測試文本: {test_text[:50]}...")
    # vec = embed_bge(test_text)
    # print(f"Embedding 向量 shape: {vec.shape}")
    # print(f"Embedding 前 5 維: {vec[:5]}")


    main()
