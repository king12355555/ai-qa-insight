# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import sqlite3
import json
from pathlib import Path
from openai import OpenAI
from collections import Counter
from datetime import datetime

# ---------- 知识库（在线回答 + 离线兜底 共用）----------
QA_KB = [
    {"q": "维生素D", "a": "中国营养学会建议：宝宝出生后数天就应开始补充维生素D，每天400国际单位(IU)，至少补到2岁。维生素D能促进钙吸收，缺乏可能引起佝偻病、睡眠不安、枕秃等。可单独用维生素D滴剂补充；婴幼儿皮肤娇嫩，不建议靠暴晒补D。"},
    {"q": "钙", "a": "不同年龄每日钙推荐量：0-6月约200mg、7-12月约250mg、1-3岁约600mg。补钙先补D，维生素D不足时补钙效果差。奶类、豆制品、深绿色蔬菜都是补钙好来源。"},
    {"q": "DHA", "a": "DHA是大脑和视网膜发育的重要营养，深海鱼、藻油是主要来源。建议优先食补（每周2-3次深海鱼），配方奶或辅食不足再考虑补充剂。"},
    {"q": "益生菌", "a": "益生菌有助于维持肠道菌群平衡，常见菌株有乳双歧杆菌、鼠李糖乳杆菌等。冲泡水温不要超过40℃，否则会杀灭活菌。益生菌不是越多越好，应结合腹泻、便秘、过敏等场景按需补充，必要时咨询医生。"},
    {"q": "奶粉", "a": "选奶粉主要看奶源、配方、段数（1段0-6月、2段6-12月、3段1-3岁）。关注蛋白质、DHA、益生元等核心成分，不必盲目追求越贵越好。转奶要循序渐进，避免突然更换引起肠胃不适。"},
]


def kb_text():
    return "\n".join(f"【{i['q']}】{i['a']}" for i in QA_KB)


SYSTEM_PROMPT = (
    "你是家庭健康品牌的 AI 育儿/营养顾问，服务对象是宝爸宝妈。\n"
    "要求：1) 专业、通俗、温暖，像一位懂营养的朋友；不说绝对化医疗结论，涉及疾病诊断请建议咨询医生；"
    "2) 优先用【参考知识库】回答；知识库没有的，基于可靠健康常识回答，并提示「建议咨询医生/专业营养师」；"
    "3) 结尾给1条实用小贴士。\n\n【参考知识库】\n" + kb_text()
)


def offline_answer(q):
    for i in QA_KB:
        if i["q"] in q:
            return i["a"]
    return "这个问题我暂时无法准确回答，建议咨询儿科医生或专业营养师哦。（离线兜底模式，接入大模型后可回答任意问题）"


# ---------- 主题与运营建议（离线兜底用）----------
TOPICS = ["维生素D", "钙", "DHA", "益生菌", "奶粉", "辅食", "睡眠", "皮肤", "其他/综合"]

SUGGESTIONS = {
    "维生素D": "用户最关心「维生素D补充」→ 建议内容主推婴幼儿补D科普，可搭配维生素D滴剂做社群团购。",
    "钙": "用户补钙需求高 → 建议做「补钙食谱 + 钙片选购」内容，重点主推儿童钙类产品。",
    "DHA": "用户关注 DHA → 建议内容主推「DHA 食补 vs 补充剂」，关联藻油 DHA 产品。",
    "益生菌": "用户对益生菌疑问多 → 建议做「益生菌冲泡与选购」科普，绑定肠道健康场景。",
    "奶粉": "用户选奶粉/转奶需求集中 → 建议做「奶粉段数与转奶攻略」，引导私域 1 对 1 顾问。",
    "辅食": "用户关注辅食添加 → 建议做「辅食添加顺序与食谱」内容，绑定辅食工具/食材产品。",
    "睡眠": "用户关注宝宝睡眠 → 建议做「睡眠习惯养成」科普，关联安抚类产品或付费课程。",
    "皮肤": "用户关注皮肤护理 → 建议做「湿疹/红屁屁护理」科普，推荐护理产品并引导问诊。",
    "其他/综合": "用户提问较分散 → 建议先扩充知识库覆盖更多高频问题，再沉淀稳定需求。",
}


# ---------- 服务商预设（OpenAI 兼容接口）----------
PROVIDERS = {
    "DeepSeek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "豆包（火山方舟）": {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-1-5-pro-32k-250115"},
    "Kimi（月之暗面）": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
    "通义千问（阿里云）": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "Ollama（本地）": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5"},
    "自定义": {"base_url": "", "model": ""},
}


def is_online():
    """AI 开关：开启时用大模型，关闭时走本地知识库兜底"""
    return st.session_state.get("ai_enabled", True)


def get_client():
    key = st.session_state.get("api_key", "") or "sk-local"
    base = st.session_state.get("base_url", "") or "https://api.deepseek.com"
    return OpenAI(api_key=key, base_url=base)


def get_model():
    return st.session_state.get("model", "") or "deepseek-chat"


# ---------- 配置持久化（保存 API / 模型等到本地 config.json）----------
CONFIG_PATH = Path(__file__).parent / "config.json"

CONFIG_DEFAULTS = {
    "ai_enabled": True,
    "provider": "DeepSeek",
    "base_url": PROVIDERS["DeepSeek"]["base_url"],
    "model": PROVIDERS["DeepSeek"]["model"],
    "api_key": "",
}


def load_config():
    """从 config.json 读取配置，缺失项用默认值"""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {**CONFIG_DEFAULTS, **data}
        except Exception:
            return dict(CONFIG_DEFAULTS)
    return dict(CONFIG_DEFAULTS)


def save_config():
    """把当前配置写入 config.json"""
    data = {
        "ai_enabled": st.session_state.get("ai_enabled", True),
        "provider": st.session_state.get("provider", "DeepSeek"),
        "base_url": st.session_state.get("base_url", ""),
        "model": st.session_state.get("model", ""),
        "api_key": st.session_state.get("api_key", ""),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------- 数据持久化 ----------
DB_PATH = Path(__file__).parent / "insights.db"


def init_db():
    """初始化本地数据库（离线可用，数据长期保存）"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS questions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "time TEXT, q TEXT, topic TEXT)"
    )
    conn.commit()
    conn.close()


def record(q):
    """记录每次提问，持久化到本地 SQLite"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO questions (time, q, topic) VALUES (?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), q, categorize(q)),
    )
    conn.commit()
    conn.close()


def load_log():
    """读取全部提问记录"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT time, q, topic FROM questions ORDER BY id")
    rows = [{"time": r[0], "q": r[1], "topic": r[2]} for r in cur.fetchall()]
    conn.close()
    return rows


def clear_log():
    """清空提问记录"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM questions")
    conn.commit()
    conn.close()


# ---------- AI 打标签 ----------
def keyword_categorize(q):
    """离线兜底：关键词匹配"""
    for i in QA_KB:
        if i["q"] in q:
            return i["q"]
    return "其他/综合"


def ai_categorize(q):
    """用大模型给问题打标签（在线）；失败/离线则用关键词兜底"""
    if not is_online():
        return keyword_categorize(q)
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=get_model(),
            temperature=0,
            max_tokens=20,
            messages=[
                {"role": "system", "content": f"你是母婴育儿问题分类器。请从以下主题里选最合适的一个，只返回主题名，不要解释、不要标点：{TOPICS}"},
                {"role": "user", "content": q},
            ],
        )
        topic = resp.choices[0].message.content.strip()
        return topic if topic in TOPICS else keyword_categorize(q)
    except Exception:
        return keyword_categorize(q)


def categorize(q):
    return ai_categorize(q)


# ---------- AI 生成运营建议 ----------
def ai_suggestions(log, topics, qc):
    """读统计数据 → 让大模型生成运营建议（在线）；离线则用规则兜底"""
    if not is_online():
        top_topic = topics.most_common(1)[0][0]
        return SUGGESTIONS.get(top_topic, SUGGESTIONS["其他/综合"])
    try:
        topic_str = "、".join(f"{t}（{c}次）" for t, c in topics.most_common(5))
        q_str = "；".join(f"{q}（{c}次）" for q, c in qc.most_common(5))
        client = get_client()
        resp = client.chat.completions.create(
            model=get_model(),
            temperature=0.5,
            messages=[
                {"role": "system", "content": "你是母婴/家庭健康私域运营的数据分析师。根据用户提问数据，给出3条可落地的运营建议（覆盖内容方向、选品方向、社群动作），每条一句话，用换行分隔，不要客套话和标题。"},
                {"role": "user", "content": f"累计提问 {len(log)} 次。主题分布：{topic_str}。高频问题：{q_str}。请给出运营建议。"},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        top_topic = topics.most_common(1)[0][0]
        return SUGGESTIONS.get(top_topic, SUGGESTIONS["其他/综合"])


# ---------- 用户洞察看板 ----------
def render_insights():
    st.markdown("### 📊 用户需求洞察")
    st.caption("把每次提问沉淀为数据 → 反哺选品和内容。数据已持久化到本地 insights.db，重启不丢。这是通用大模型（豆包/DeepSeek）做不到的。")
    log = load_log()
    if not log:
        st.info("还没有提问记录，先去「💬 智能答疑」页点几个快捷问题试试。")
        return

    total = len(log)
    topics = Counter(x["topic"] for x in log)
    qc = Counter(x["q"] for x in log)

    col1, col2 = st.columns(2)
    col1.metric("累计提问", total)
    col2.metric("覆盖主题", len(topics))

    st.markdown("**用户关注主题分布**")
    df = pd.DataFrame({"提问数": list(topics.values())}, index=list(topics.keys()))
    st.bar_chart(df)

    st.markdown("**高频问题 TOP**")
    for i, (q, c) in enumerate(qc.most_common(5), 1):
        st.write(f"{i}. {q}（{c} 次）")

    label = "运营建议（AI 生成）" if is_online() else "运营建议（规则兜底）"
    st.markdown(f"**{label}**")
    st.success(ai_suggestions(log, topics, qc))

    # 清空数据（二次确认）
    if "confirm_clear" not in st.session_state:
        st.session_state["confirm_clear"] = False

    if st.session_state["confirm_clear"]:
        st.warning("⚠️ 确定要清空所有提问记录吗？此操作不可恢复。")
        c1, c2 = st.columns(2)
        if c1.button("确认清空", type="primary"):
            clear_log()
            st.session_state["confirm_clear"] = False
            st.rerun()
        if c2.button("取消"):
            st.session_state["confirm_clear"] = False
            st.rerun()
    else:
        if st.button("🧹 清空数据"):
            st.session_state["confirm_clear"] = True
            st.rerun()


# ---------- 界面 ----------
init_db()

# 加载已保存的配置（API / 模型等）
if "config_loaded" not in st.session_state:
    cfg = load_config()
    for k, v in cfg.items():
        st.session_state[k] = v
    st.session_state["prev_provider"] = cfg.get("provider", "DeepSeek")
    st.session_state["config_loaded"] = True

st.set_page_config(page_title="母婴健康 AI 答疑助手", page_icon="👶")
st.title("👶 母婴健康 AI 答疑助手")

with st.sidebar:
    st.markdown("### ⚙️ 设置")
    st.toggle("启用 AI 模式", key="ai_enabled",
              help="开关：开启用大模型，关闭走本地知识库兜底（方便现场对比演示）")

    provider = st.selectbox("服务商", list(PROVIDERS.keys()), key="provider")
    if st.session_state.get("prev_provider") != provider:
        st.session_state["prev_provider"] = provider
        st.session_state["base_url"] = PROVIDERS[provider]["base_url"]
        st.session_state["model"] = PROVIDERS[provider]["model"]

    st.text_input("API 地址 (Base URL)", key="base_url")
    st.text_input("模型 (Model)", key="model")
    st.text_input("API Key", type="password", key="api_key", help="本地 Ollama 可不填")

    status = "🟢 AI 在线" if is_online() else "⚪ 离线兜底"
    st.caption(f"当前状态：{status} · 配置已自动保存到本地 config.json")
    save_config()

tab_chat, tab_insight = st.tabs(["💬 智能答疑", "📊 用户洞察"])

with tab_chat:
    st.markdown("**点一个试试 👇**")
    quick = [
        "宝宝多大开始补维生素D？每天补多少？",
        "一岁宝宝要不要额外补钙？",
        "益生菌冲泡水温多少合适？",
        "DHA有必要额外补吗？",
        "给宝宝选奶粉主要看什么？",
        "宝宝六个月可以加什么辅食？",
    ]
    cols = st.columns(len(quick))
    for i, q in enumerate(quick):
        if cols[i].button(q, key=f"q{i}"):
            st.session_state["pending"] = q

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if st.session_state["messages"]:
        if st.button("🧹 清空对话"):
            st.session_state["messages"] = []
            st.rerun()

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("输入你的育儿/营养问题…")
    prompt = prompt or st.session_state.pop("pending", None)

    if prompt:
        record(prompt)
        st.session_state["messages"].append({"role": "user", "content": prompt})

        answer = None
        if is_online():
            try:
                client = get_client()
                with st.spinner("思考中…"):
                    resp = client.chat.completions.create(
                        model=get_model(),
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state["messages"],
                    )
                answer = resp.choices[0].message.content
            except Exception:
                answer = None
        if answer is None:
            answer = offline_answer(prompt)

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.rerun()

with tab_insight:
    render_insights()
