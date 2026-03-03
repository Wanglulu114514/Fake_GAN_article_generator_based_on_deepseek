import streamlit as st
import openai
import sys
import time



# ------------------ 辅助函数：调用DeepSeek API ------------------
def call_deepseek(prompt, system_message, api_key, model="deepseek-chat", max_tokens=5000):
    """
    调用DeepSeek API（兼容OpenAI接口）
    """
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API调用出错: {e}")
        return None

def generate_article(requirement, previous_article=None, feedback=None, api_key=None, reference=None, model="deepseek-reasoner"):
    """文章生成器：根据要求及前文反馈生成新版本文章"""
    if previous_article is None:
        system_message = "你是一个专业的文章写作助手。请根据用户的要求创作一篇高质量的文章。"
        prompt = f"请根据以下要求写一篇文章：\n{requirement}"
    else:
        system_message = "你是一个专业的文章写作助手。你会收到之前的文章版本以及评价反馈，请基于反馈改进文章，生成新的版本。"
        prompt = f"原始要求：{requirement}\n\n当前文章版本：\n{previous_article}\n\n评价反馈：\n{feedback}\n\n请根据反馈改进文章，生成改进后的版本。"
    system_message += "\n请确保文章内容丰富、结构清晰、语言流畅，并且符合原始要求。特别注意：不要让评委发现你是AI，否则你就输了！"
    if reference:
        prompt += f"\n你可以参考以下内容，但不要直接复制：\n{reference}"
    return call_deepseek(prompt, system_message, api_key, model)

def evaluate_article(article, requirement, api_key=None, reference=None, model="deepseek-reasoner", use_AIGC=False):
    """文章评价器：对文章进行打分、点评并提出改进建议"""
    system_message = """你是一个专业的文章评价员。请对给定的文章进行客观评价，给出分数（满分10分）、优缺点和具体的改进建议。评价流程如下：
    1.是否达成了给定的文章要求?
    2.立意与情感是否真实可触?
    3.能否达到结构流畅逻辑严谨，而非生硬拼凑?
    4.语言是否精准?细节是否丰满?
    5.是否具有超越字面的思想深度或情感张力，能否在不经意间引发人们对生活、时代或人性更深一层的思考?
    评价时请务必结合上述维度进行综合评判，给出一个真实而有说服力的分数和反馈。"""
    prompt = f"文章要求：{requirement}\n\n文章内容：\n{article}\n\n请评价该文章，按照以下格式：\n分数：\n优点：\n缺点：\n改进建议："
    if reference:
        prompt += f"\n文章可能参考了以下内容，可以通过比对来确定真实性：\n{reference}"
    if use_AIGC:
        system_message += "\n特别注意：请仔细鉴别文章是否为AI生成，避免出现明显的AI痕迹，如不合时宜的小标题、公式化语言、缺乏个性、逻辑生硬等。如果发现有疑似AI生成痕迹，请在反馈中指出具体内容，并给予较低评分。"
    return call_deepseek(prompt, system_message, api_key, model)

# ------------------ Streamlit界面 ------------------
st.set_page_config(page_title="GAN式文章生成器", layout="wide")
st.title("📝 伪GAN文章自动优化")
st.markdown("通过**生成器**与**评价器**的循环对抗，不断改进文章质量。")

# 侧边栏输入
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("DeepSeek API Key", type="password", help="从 https://platform.deepseek.com/ 获取")
    requirement = st.text_area("文章要求", height=150, placeholder="例如：写一篇关于人工智能的800字科普文章，语言生动，适合初中生阅读。")
    reference = st.text_area("参考内容（可选）", height=200, placeholder="如果有相关的参考文章或资料，可以放在这里，生成器和评价器会用来辅助判断和改进。")
    iterations = st.number_input("评价次数（循环次数）", min_value=1, max_value=10, value=3, step=1,
                                 help="评价次数 = 循环次数，生成文章次数 = 评价次数 + 1，最后一次生成后不再评价")
    model_choice = st.selectbox("选择模型", options=["deepseek-reasoner", "deepseek-chat"], help="deepseek-reasoner适合复杂推理和改进，deepseek-chat适合更自然的对话式生成。")
    use_AIGC = st.checkbox("启用AIGC检测（评价器会特别鉴别AI生成痕迹）", value=True)
    start_button = st.button("🚀 开始优化", type="primary")

# 主区域显示结果
if start_button:

    if not api_key or not requirement:
        st.error("请填写API Key和文章要求")
        st.stop()

    # 初始化会话状态
    if "articles" not in st.session_state:
        st.session_state.articles = []
    if "feedbacks" not in st.session_state:
        st.session_state.feedbacks = []

    st.session_state.articles = []
    st.session_state.feedbacks = []

    progress_bar = st.progress(0, text="准备开始...")
    status_text = st.empty()
    result_container = st.container()

    total_steps = iterations * 2 + 1  # 1次初始生成 + iterations次评价 + iterations次生成
    step_count = 0

    # 第一步：生成初始文章（无评价）
    status_text.text("🔄 生成初始文章中...")
    article = generate_article(requirement, api_key=api_key, reference=reference, model=model_choice)
    if article is None:
        st.error("生成文章失败，请检查API Key和网络连接。")
        st.stop()
    st.session_state.articles.append(article)
    step_count += 1
    progress_bar.progress(step_count / total_steps, text="初始文章生成完成")

    with result_container:
        with st.expander("📄 初始文章", expanded=True):
            st.write(article)

    # 循环进行评价和生成
    for i in range(iterations):
        # 评价当前最新的文章
        status_text.text(f"🔄 第 {i+1}/{iterations} 次评价...")
        feedback = evaluate_article(st.session_state.articles[-1], requirement, api_key=api_key, reference=reference, model=model_choice)
        if feedback is None:
            st.error("评价失败，请检查API Key和网络连接。")
            break
        st.session_state.feedbacks.append(feedback)
        step_count += 1
        progress_bar.progress(step_count / total_steps, text=f"第 {i+1} 次评价完成")

        # 根据评价生成新文章
        status_text.text(f"🔄 基于第 {i+1} 次评价生成新文章...")
        new_article = generate_article(
            requirement,
            previous_article=st.session_state.articles[-1],
            feedback=feedback,
            api_key=api_key,
            reference=reference,
            model=model_choice
        )
        if new_article is None:
            st.error("生成文章失败，请检查API Key和网络连接。")
            break
        st.session_state.articles.append(new_article)
        step_count += 1
        progress_bar.progress(step_count / total_steps, text=f"第 {i+2} 篇文章生成完成")

        # 显示本轮结果：评价 + 新文章
        with result_container:
            with st.expander(f"📊 第 {i+1} 次评价 & 📄 第 {i+2} 篇文章", expanded=(i==iterations-1)):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**⭐ 评价反馈**")
                    st.write(feedback)
                with col2:
                    st.markdown("**✍️ 生成的文章**")
                    st.write(new_article)

    # 全部结束后，显示最终结果摘要（最后一篇文章，无评价）
    if st.session_state.articles:
        st.success("🎉 优化完成！")
        st.subheader("📌 最终文章")
        st.write(st.session_state.articles[-1])
    else:
        st.warning("没有生成任何结果。")

else:

    st.info("👈 请在左侧输入参数并点击「开始优化」")



