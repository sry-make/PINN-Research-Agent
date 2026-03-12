import streamlit as st
# 直接复用咱们昨天写好的硬核 RAG 逻辑
from rag_agent import ask_rag_agent 

# 1. 设置页面全局配置
st.set_page_config(
    page_title="PINN AI4S 科研助手",
    page_icon="⚛️",
    layout="centered"
)

st.title("⚛️ PINN 前沿文献检索增强助手")
st.markdown("基于本地 Qwen2.5-7B 与 ChromaDB 向量检索构建。专业解答物理信息神经网络 (PINN) 与计算力学问题。")
st.divider()

# 2. 初始化聊天历史记录 (Session State 用于保持页面刷新时不丢失记忆)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是您的专属 AI4S 科研助手，我已经阅读了您的本地知识库。请问有什么关于 PINN 或文献细节的硬核问题需要我解答？"}
    ]

# 3. 在界面上渲染之前的聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 监听用户的聊天输入
if prompt := st.chat_input("例如：Raissi 2019年论文中的损失函数是如何构造的？"):
    
    # 将用户的问题显示在界面上并存入历史记录
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示模型正在思考的动画
    with st.chat_message("assistant"):
        with st.spinner("🧠 正在检索本地论文库并进行推理..."):
            # 调用你的核心 RAG 函数！
            response = ask_rag_agent(prompt)
            # 将大模型的回答渲染到网页上 (Streamlit 会自动把 LaTeX 代码渲染成漂亮的高等数学公式)
            st.markdown(response)
            
    # 将助手的回答存入历史记录
    st.session_state.messages.append({"role": "assistant", "content": response})