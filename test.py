"""
GPT‑4o mini vs OSS（Ollama）模型比較工具
-------------------------------------------------
功能摘要：
1) 同一題庫，並行測 GPT‑4o mini（Azure OpenAI）與 OSS（Ollama OpenAI‑compatible API）。
2) 規範化 Prompt、記錄回應、量測延遲（Latency）。
3) 指標：正確率、格式符合率（JSON 類題）、平均延遲；保留質化觀察欄位。
4) 類別覆蓋：邏輯推理、事實問答、長文本理解、多輪對話、格式輸出。
5) 以 .env 管理金鑰與端點。

使用方式：
1) 建立虛擬環境並安裝依賴（requests、pandas、python-dotenv、openai、tqdm）。
2) 於專案根目錄建立 .env（見最下方範例）。
3) 直接執行：python gpt4o-mini_vs_oss_comparator.py
   產出：results/results.csv 與 results/summary.csv

注意：
- 內建僅放少量示例題；實測請自行擴充「題庫區」清單（建議：邏輯20、事實20、長文10、多輪5組、格式5）。
- Azure 需使用 AzureOpenAI 客戶端；Ollama 使用 OpenAI 相容端點（/v1）。
"""

import os
import json
import time
import math
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm
import requests

# -------------------------------
# 0. 讀取設定
# -------------------------------
load_dotenv()

print("DEBUG: AOAI_KEY =", os.getenv("AOAI_KEY"))
print("DEBUG: AOAI_ENDPOINT =", os.getenv("AOAI_ENDPOINT"))
print("DEBUG: AOAI_API_VERSION =", os.getenv("AOAI_API_VERSION"))
print("DEBUG: AOAI_CHAT_DEPLOYMENT =", os.getenv("AOAI_CHAT_DEPLOYMENT"))
print("DEBUG: OLLAMA_API_BASE =", os.getenv("OLLAMA_API_BASE"))
print("DEBUG: OLLAMA_API_KEY =", os.getenv("OLLAMA_API_KEY"))
print("DEBUG: OLLAMA_MODEL =", os.getenv("OLLAMA_MODEL"))

QUESTIONS_JSON = os.getenv("QUESTIONS_JSON", "questions.json")  

# Azure OpenAI（GPT‑4o mini）
AOAI_KEY = os.getenv("AOAI_KEY", "")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT", "")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION", "2024-10-21-preview")
AOAI_CHAT_DEPLOYMENT = os.getenv("AOAI_CHAT_DEPLOYMENT", "gpt-4o-mini")  # 部署名稱（等於 model 參數）

# Ollama（OpenAI‑compatible）
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")  # 佔位字串即可
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

# I/O
RESULT_DIR = Path("results"); RESULT_DIR.mkdir(exist_ok=True)
RESULT_CSV = RESULT_DIR / "results.csv"
SUMMARY_CSV = RESULT_DIR / "summary.csv"

# -------------------------------
# 1. 客戶端初始化
# -------------------------------
if not AOAI_KEY or not AOAI_ENDPOINT:
    print("⚠️  警告：未偵測到 AOAI_KEY / AOAI_ENDPOINT，Azure 端可能無法請求。")

client_azure = AzureOpenAI(
    api_key=AOAI_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version=AOAI_API_VERSION,
)

client_ollama = OpenAI(
    api_key=OLLAMA_API_KEY,
    base_url=OLLAMA_API_BASE,
)

# -------------------------------
# 2. Prompt 樣板
# -------------------------------
PROMPT_REASONING = (
    "你是一個專業的推理助手，請根據以下條件回答問題：\n"
    "問題：{question}\n"
    "請逐步推導並得出最終答案。"
)

PROMPT_FACT = (
    "你是一個精準的知識助手。\n"
    "請對下列問題給出簡潔且明確的答案。\n"
    "問題：{question}"
)

PROMPT_LONG = (
    "請閱讀以下文章，再回答問題。\n\n文章：\n{passage}\n\n問題：{question}\n"
    "請引用文中關鍵句並作答。"
)

PROMPT_FORMAT_JSON = (
    "請嚴格以 JSON 輸出並符合鍵名與型別。\n"
    "問題：{question}\n"
    "輸出格式：{{\"answer\": <字串>, \"confidence\": <0~1數字>}}\n"
    "只輸出 JSON，不要多餘文字。"
)

# 多輪對話：每組是一個 messages list 的骨架；最後加上 user 問句
MULTI_TURN_SYSTEM = {
    "role": "system",
    "content": "你是一位耐心而嚴謹的助理，會記住先前設定與事實，並在後續回答中維持一致。",
}

# -------------------------------
# 3. 題庫區（完整版）
# -------------------------------

# ❶ 邏輯推理（20 題）
logic_questions = [
    {"id": "L1",  "question": "小明有 7 顆糖，送了 3 顆，又買了 2 顆。現在有幾顆？", "answer": "6"},
    {"id": "L2",  "question": "如果今天是星期一，100 天後是星期幾？（以 7 天為一週）", "answer": "星期三"},
    {"id": "L3",  "question": "一個長方形長 12、寬 5，面積是多少？", "answer": "60"},
    {"id": "L4",  "question": "最小的正整數 n 能被 1~10 全整除，n 是多少？", "answer": "2520"},
    {"id": "L5",  "question": "一數列為 1, 1, 2, 3, 5, 8, 下一項是多少？", "answer": "13"},
    {"id": "L6",  "question": "3x ≡ 2 (mod 11)，x 最小非負解？", "answer": "8"},
    {"id": "L7",  "question": "12 與 18 的最小公倍數？", "answer": "36"},
    {"id": "L8",  "question": "時鐘 3:30 時，時針與分針夾角最小值（度）？", "answer": "75"},
    {"id": "L9",  "question": "從 2,6,14,30 依規律續一項為？", "answer": "62"},  # +4,+8,+16,+32
    {"id": "L10", "question": "來回等距旅程，去程 60km/h、回程 40km/h，平均速率？", "answer": "48"},
    {"id": "L11", "question": "A 單獨 6 小時完成，B 單獨 4 小時完成，合作需幾小時？", "answer": "2.4"},
    {"id": "L12", "question": "2 的 20 次方除以 7 的餘數？", "answer": "4"},
    {"id": "L13", "question": "前 50 個偶數和（2+4+...+100）是多少？", "answer": "2550"},
    {"id": "L14", "question": "用 50、10、5、1 元最少幾枚湊 93 元？", "answer": "8"},
    {"id": "L15", "question": "一數列 3,7,15,31，下一項？", "answer": "63"},  # *2+1
    {"id": "L16", "question": "某班 40 人，男生 18，女生佔全班幾人？", "answer": "22"},
    {"id": "L17", "question": "有 24 支鉛筆，平均分給 7 人，每人幾支、剩幾支？（格式：a,b）", "answer": "3,3"},
    {"id": "L18", "question": "一數的 30% 是 45，該數為？", "answer": "150"},
    {"id": "L19", "question": "兩數相差 9，和為 41，較大者為？", "answer": "25"},
    {"id": "L20", "question": "一商品原價 500，先打 8 折再打 9 折，最終價格？", "answer": "360"},
]

# ❷ 事實性問答（20 題）— 避免時效性，選穩定常識
fact_questions = [
    {"id": "F1",  "question": "水在標準大氣壓下的沸點（°C）？", "answer": "100"},
    {"id": "F2",  "question": "地球有幾個大洲？", "answer": "7"},
    {"id": "F3",  "question": "一分鐘等於幾秒？", "answer": "60"},
    {"id": "F4",  "question": "π（圓周率）以兩位小數近似為？", "answer": "3.14"},
    {"id": "F5",  "question": "人類有幾對染色體？", "answer": "23"},
    {"id": "F6",  "question": "光速約為每秒多少公里？（取整數）", "answer": "300000"},
    {"id": "F7",  "question": "氧氣的化學符號？", "answer": "O2"},
    {"id": "F8",  "question": "鹽（食鹽）的主要成分化學式？", "answer": "NaCl"},
    {"id": "F9",  "question": "攝氏零度相當於華氏幾度？（四捨五入到整數）", "answer": "32"},
    {"id": "F10", "question": "一年平年有幾天？", "answer": "365"},
    {"id": "F11", "question": "世界上使用人數最多的母語大致是哪一語言家族的語言？（簡答）", "answer": "中文"},
    {"id": "F12", "question": "金的化學符號？", "answer": "Au"},
    {"id": "F13", "question": "人體主要吸入的氣體中，氮氣約佔體積百分比？（整數）", "answer": "78"},
    {"id": "F14", "question": "DNA 的全名英文縮寫意指？（簡答：可填 “Deoxyribonucleic acid”）", "answer": "Deoxyribonucleic acid"},
    {"id": "F15", "question": "地球自轉一圈大約為幾小時？", "answer": "24"},
    {"id": "F16", "question": "太陽系中體積最大的行星？", "answer": "木星"},
    {"id": "F17", "question": "攝氏與華氏在何溫度相等？（°C）", "answer": "-40"},
    {"id": "F18", "question": "公制長度單位中，1 公尺等於幾公分？", "answer": "100"},
    {"id": "F19", "question": "海平面上標準大氣壓約等於幾百帕（hPa）？（整數）", "answer": "1013"},
    {"id": "F20", "question": "純水在 4°C 時的密度（g/cm^3）約為？", "answer": "1"},
]

# ❸ 長文本理解（10 題）— 每題含短文 + 問題 + 應包含的關鍵片語
long_questions = [
    {
        "id": "T1",
        "passage": "在資料科學中，特徵工程是提升模型表現的關鍵步驟。透過缺值補齊、標準化與類別編碼，能讓訓練更穩定，並縮短收斂時間。",
        "question": "文中提到特徵工程的目的為何？",
        "answer_contains": "提升模型表現",
    },
    {
        "id": "T2",
        "passage": "雲端運算提供彈性的運算資源，使用者可以按需擴展。這降低了前期硬體投資，並且讓服務更容易全球佈署。",
        "question": "雲端運算帶來的兩個主要優勢是什麼？",
        "answer_contains": "彈性的運算資源",
    },
    {
        "id": "T3",
        "passage": "版本控制系統如 Git，讓多人協作能夠追蹤變更、分支開發並進行合併。良好的提交訊息有助於回溯問題。",
        "question": "版本控制系統的其中一個功能是協助什麼？",
        "answer_contains": "追蹤變更",
    },
    {
        "id": "T4",
        "passage": "在專案管理中，敏捷方法強調短週期迭代、持續回饋，以及跨職能團隊的密切合作，以快速回應需求變更。",
        "question": "敏捷方法強調的節奏為何？",
        "answer_contains": "短週期迭代",
    },
    {
        "id": "T5",
        "passage": "資料可視化能將複雜數據轉為圖形，幫助人們快速找出趨勢與異常點。選擇合適圖表類型相當重要。",
        "question": "資料可視化的其中一個好處是什麼？",
        "answer_contains": "快速找出趨勢",
    },
    {
        "id": "T6",
        "passage": "在軟體測試中，單元測試專注於最小可測單元；整合測試則關注模組之間的互動，確保系統整體行為正確。",
        "question": "整合測試關注的重點是什麼？",
        "answer_contains": "模組之間的互動",
    },
    {
        "id": "T7",
        "passage": "資料庫索引能加速查詢，但會增加寫入成本與儲存空間。建立索引前應評估查詢模式與更新頻率。",
        "question": "建立索引的代價之一是什麼？",
        "answer_contains": "增加寫入成本",
    },
    {
        "id": "T8",
        "passage": "機器學習模型若在訓練集表現良好但在測試集表現差，稱為過擬合。常見緩解方法包括正則化與資料增強。",
        "question": "過擬合可用何種方法緩解？",
        "answer_contains": "正則化",
    },
    {
        "id": "T9",
        "passage": "多執行緒可以提升 I/O 密集工作效率，但在 CPU 密集工作時可能受限於 GIL 或上下文切換開銷。",
        "question": "何種情境下多執行緒較能提升效率？",
        "answer_contains": "I/O 密集",
    },
    {
        "id": "T10",
        "passage": "快取是一種以空間換取時間的策略，將常用資料暫存可降低延遲，但需要設計失效策略以確保資料新鮮度。",
        "question": "使用快取需特別注意何事項？",
        "answer_contains": "失效策略",
    },
]

# ❹ 多輪對話（5 組）— 測記憶與一致性
MULTI_TURN_SYSTEM = {
    "role": "system",
    "content": "你是一位耐心而嚴謹的助理，會記住先前設定與事實，並在後續回答中維持一致。"
}

multi_turn_sets = [
    {
        "id": "C1",
        "context": [
            MULTI_TURN_SYSTEM,
            {"role": "user", "content": "我叫小藍，之後請用我的名字回覆。"},
            {"role": "assistant", "content": "好的，小藍！我會記住你的名字。"},
        ],
        "follow_up": {"role": "user", "content": "提醒我一下我剛剛說我叫什麼？"},
        "expected_contains": "小藍",
    },
    {
        "id": "C2",
        "context": [
            MULTI_TURN_SYSTEM,
            {"role": "user", "content": "我最喜歡的顏色是綠色，請記住。"},
            {"role": "assistant", "content": "明白了，你最喜歡綠色。"},
        ],
        "follow_up": {"role": "user", "content": "我剛剛說我喜歡哪個顏色？"},
        "expected_contains": "綠色",
    },
    {
        "id": "C3",
        "context": [
            MULTI_TURN_SYSTEM,
            {"role": "user", "content": "我今天 15:00 有會議，之後問我開會時間要一致。"},
            {"role": "assistant", "content": "收到，今天 15:00。"},
        ],
        "follow_up": {"role": "user", "content": "我的會議幾點？"},
        "expected_contains": "15:00",
    },
    {
        "id": "C4",
        "context": [
            MULTI_TURN_SYSTEM,
            {"role": "user", "content": "之後請用敬語稱呼我為您。"},
            {"role": "assistant", "content": "好的，之後我會以您稱呼。"},
        ],
        "follow_up": {"role": "user", "content": "請再提醒我你的稱呼方式需要注意什麼？"},
        "expected_contains": "您",
    },
    {
        "id": "C5",
        "context": [
            MULTI_TURN_SYSTEM,
            {"role": "user", "content": "假設台北是出發地，目的地是台中。"},
            {"role": "assistant", "content": "了解，從台北前往台中。"},
        ],
        "follow_up": {"role": "user", "content": "剛剛設定的出發地是哪裡？"},
        "expected_contains": "台北",
    },
]

# ❺ JSON/格式輸出（5 題）— 嚴格要求 JSON
json_format_tasks = [
    {"id": "J1", "question": "用一句話解釋機器學習與深度學習的差異。"},
    {"id": "J2", "question": "以一句話給出軟體測試中單元測試的定義。"},
    {"id": "J3", "question": "以一句話描述資料庫索引的優缺點各一。"},
    {"id": "J4", "question": "以一句話說明快取（cache）的核心目的。"},
    {"id": "J5", "question": "以一句話描述敏捷開發中迭代（iteration）的意義。"},
]

def load_questions_from_json(path: str) -> bool:
    """
    從外部 JSON 檔載入題庫。若檔案存在且格式正確，就覆蓋內建清單；
    若不存在或讀取失敗，維持原內建題庫不變。
    JSON 結構需包含任意下列 key（皆為 list）：
      "logic", "fact", "longtext", "multiturn", "jsonformat"
    """
    p = Path(path)
    if not p.exists():
        print(f"ℹ️  未找到外部題庫：{path}，使用內建題庫。")
        return False
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        def pick(name, default):
            return data.get(name, default) if isinstance(data.get(name, None), list) else default

        # 覆蓋內建題庫（若對應 key 存在）
        global logic_questions, fact_questions, long_questions, multi_turn_sets, json_format_tasks
        logic_questions      = pick("logic",      logic_questions)
        fact_questions       = pick("fact",       fact_questions)
        long_questions       = pick("longtext",   long_questions)
        multi_turn_sets      = pick("multiturn",  multi_turn_sets)
        json_format_tasks    = pick("jsonformat", json_format_tasks)

        print(f"✅ 已載入外部題庫：{path}")
        return True
    except Exception as e:
        print(f"⚠️  載入外部題庫失敗（{path}）：{e}，使用內建題庫。")
        return False

# 立刻嘗試載入（若檔案不存在就安靜使用內建題庫）
_ = load_questions_from_json(QUESTIONS_JSON)


# -------------------------------
# 4. 請求與評分工具
# -------------------------------

def ask_azure(messages: List[Dict[str, str]], temperature: float = 0.0) -> Tuple[str, float]:
    """呼叫 Azure OpenAI（GPT‑4o mini）並回傳 (文字, 秒數)。"""
    t0 = time.perf_counter()
    resp = client_azure.chat.completions.create(
        model=AOAI_CHAT_DEPLOYMENT,  # 在 Azure SDK 裡 model=部署名稱
        messages=messages,
        temperature=temperature,
    )
    dt = time.perf_counter() - t0
    content = resp.choices[0].message.content.strip()
    return content, dt


def ask_ollama(messages: List[Dict[str, str]], temperature: float = 0.0) -> Tuple[str, float]:
    """呼叫 Ollama（OpenAI‑compatible /v1）並回傳 (文字, 秒數)。"""
    t0 = time.perf_counter()
    resp = client_ollama.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        temperature=temperature,
    )
    dt = time.perf_counter() - t0
    content = resp.choices[0].message.content.strip()
    return content, dt


def is_number_equal(pred: str, gold: str) -> bool:
    """嘗試將字串中的整數/浮點數抓出來比對；若金標非數字則退回純字串相等。"""
    def to_num(x: str):
        try:
            return float(x.replace(",", "").strip())
        except:
            return None
    pn, gn = to_num(pred), to_num(gold)
    if pn is not None and gn is not None:
        return math.isclose(pn, gn, rel_tol=1e-6, abs_tol=1e-9)
    return pred.strip() == gold.strip()


def contains_text(pred: str, needle: str) -> bool:
    return needle in pred


def check_json_format(pred: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        obj = json.loads(pred)
        ok = isinstance(obj, dict) and "answer" in obj and "confidence" in obj
        # 進一步檢查型別
        if ok and not isinstance(obj.get("answer", None), str):
            ok = False
        if ok:
            conf = obj.get("confidence")
            ok = isinstance(conf, (int, float)) and 0 <= float(conf) <= 1
        return ok, obj if ok else {}
    except Exception:
        return False, {}

# -------------------------------
# 5. 執行各題型測試
# -------------------------------

def run_logic() -> List[Dict[str, Any]]:
    rows = []
    for item in tqdm(logic_questions, desc="Logic"):
        qid = item["id"]
        question = item["question"]
        gold = item["answer"]
        messages = [{"role": "user", "content": PROMPT_REASONING.format(question=question)}]

        a_txt, a_dt = ask_azure(messages)
        o_txt, o_dt = ask_ollama(messages)

        rows.append({
            "id": qid, "category": "logic", "model": "azure", "latency_s": round(a_dt, 3),
            "answer": a_txt, "correct": int(is_number_equal(a_txt, gold) or contains_text(a_txt, gold)),
            "format_ok": 1, "notes": ""
        })
        rows.append({
            "id": qid, "category": "logic", "model": "oss", "latency_s": round(o_dt, 3),
            "answer": o_txt, "correct": int(is_number_equal(o_txt, gold) or contains_text(o_txt, gold)),
            "format_ok": 1, "notes": ""
        })
    return rows


def run_fact() -> List[Dict[str, Any]]:
    rows = []
    for item in tqdm(fact_questions, desc="Fact"):
        qid = item["id"]
        question = item["question"]
        gold = item["answer"]
        messages = [{"role": "user", "content": PROMPT_FACT.format(question=question)}]

        a_txt, a_dt = ask_azure(messages)
        o_txt, o_dt = ask_ollama(messages)

        rows.append({
            "id": qid, "category": "fact", "model": "azure", "latency_s": round(a_dt, 3),
            "answer": a_txt, "correct": int(is_number_equal(a_txt, gold) or contains_text(a_txt, gold)),
            "format_ok": 1, "notes": ""
        })
        rows.append({
            "id": qid, "category": "fact", "model": "oss", "latency_s": round(o_dt, 3),
            "answer": o_txt, "correct": int(is_number_equal(o_txt, gold) or contains_text(o_txt, gold)),
            "format_ok": 1, "notes": ""
        })
    return rows


def run_long() -> List[Dict[str, Any]]:
    rows = []
    for item in tqdm(long_questions, desc="LongText"):
        qid = item["id"]
        passage = item["passage"]
        question = item["question"]
        needle = item["answer_contains"]
        messages = [{"role": "user", "content": PROMPT_LONG.format(passage=passage, question=question)}]

        a_txt, a_dt = ask_azure(messages)
        o_txt, o_dt = ask_ollama(messages)

        rows.append({
            "id": qid, "category": "longtext", "model": "azure", "latency_s": round(a_dt, 3),
            "answer": a_txt, "correct": int(contains_text(a_txt, needle)),
            "format_ok": 1, "notes": ""
        })
        rows.append({
            "id": qid, "category": "longtext", "model": "oss", "latency_s": round(o_dt, 3),
            "answer": o_txt, "correct": int(contains_text(o_txt, needle)),
            "format_ok": 1, "notes": ""
        })
    return rows


def run_multi_turn() -> List[Dict[str, Any]]:
    rows = []
    for item in tqdm(multi_turn_sets, desc="MultiTurn"):
        qid = item["id"]
        base_ctx = item["context"]
        follow = item["follow_up"]
        expected = item["expected_contains"]

        # Azure
        a_msgs = list(base_ctx) + [follow]
        a_txt, a_dt = ask_azure(a_msgs)

        # OSS
        o_msgs = list(base_ctx) + [follow]
        o_txt, o_dt = ask_ollama(o_msgs)

        rows.append({
            "id": qid, "category": "multiturn", "model": "azure", "latency_s": round(a_dt, 3),
            "answer": a_txt, "correct": int(contains_text(a_txt, expected)),
            "format_ok": 1, "notes": ""
        })
        rows.append({
            "id": qid, "category": "multiturn", "model": "oss", "latency_s": round(o_dt, 3),
            "answer": o_txt, "correct": int(contains_text(o_txt, expected)),
            "format_ok": 1, "notes": ""
        })
    return rows


def run_json_tasks() -> List[Dict[str, Any]]:
    rows = []
    for item in tqdm(json_format_tasks, desc="JSONFormat"):
        qid = item["id"]
        question = item["question"]
        messages = [{"role": "user", "content": PROMPT_FORMAT_JSON.format(question=question)}]

        a_txt, a_dt = ask_azure(messages)
        a_ok, _ = check_json_format(a_txt)
        o_txt, o_dt = ask_ollama(messages)
        o_ok, _ = check_json_format(o_txt)

        rows.append({
            "id": qid, "category": "json", "model": "azure", "latency_s": round(a_dt, 3),
            "answer": a_txt, "correct": int(a_ok), "format_ok": int(a_ok), "notes": ""
        })
        rows.append({
            "id": qid, "category": "json", "model": "oss", "latency_s": round(o_dt, 3),
            "answer": o_txt, "correct": int(o_ok), "format_ok": int(o_ok), "notes": ""
        })
    return rows

# -------------------------------
# 6. 匯總與輸出
# -------------------------------

def summarize_and_save(rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")

    # 依 model+category 聚合
    grp = df.groupby(["model", "category"]).agg(
        avg_latency_s=("latency_s", "mean"),
        accuracy=("correct", "mean"),
        format_ok_rate=("format_ok", "mean"),
        n=("id", "count")
    ).reset_index()

    # 轉百分比與四捨五入
    grp["accuracy"] = (grp["accuracy"] * 100).round(1)
    grp["format_ok_rate"] = (grp["format_ok_rate"] * 100).round(1)
    grp["avg_latency_s"] = grp["avg_latency_s"].round(3)

    grp.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print("\n===== Summary =====")
    print(grp.to_string(index=False))
    print(f"\n📄 明細：{RESULT_CSV}")
    print(f"📊 摘要：{SUMMARY_CSV}")

# -------------------------------
# 7. 主程式
# -------------------------------

def main():
    all_rows: List[Dict[str, Any]] = []

    # 任務表現面
    all_rows += run_logic()
    all_rows += run_fact()
    all_rows += run_long()

    # 使用者體驗面（多輪一致性）
    all_rows += run_multi_turn()

    # 指令遵守 / 格式輸出
    all_rows += run_json_tasks()

    summarize_and_save(all_rows)

if __name__ == "__main__":
    main()

"""
.env 範例：
--------------------------------
# Azure OpenAI
AOAI_KEY=你的_AzureOpenAI_Key
AOAI_ENDPOINT=https://你的資源名稱.openai.azure.com/
AOAI_API_VERSION=2024-10-21-preview
AOAI_DEPLOYMENT=gpt-4o-mini   # 你的部署名稱

# Ollama OpenAI‑compatible
OLLAMA_API_BASE=http://127.0.0.1:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=llama3:8b
--------------------------------
擴充題庫：
- 將 logic_questions / fact_questions / long_questions / multi_turn_sets / json_format_tasks 依建議數量擴充。
- 若要外部檔案管理，可自訂 loader（讀取 JSON/CSV）後填充上述清單。
"""
