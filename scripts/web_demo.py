import random
import re
import os
import sys

from threading import Thread

import numpy as np
import streamlit as st

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
""", unsafe_allow_html=True)

system_prompt = []
device = "cuda" if torch.cuda.is_available() else "cpu"


def process_assistant_content(content):
    if model_source != "API" and 'R1' not in MODEL_PATHS[selected_model][1]:
        return content

    if '<think>' in content and '</think>' in content:
        content = re.sub(r'(<think>)(.*?)(</think>)',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\2</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' in content and '</think>' not in content:
        content = re.sub(r'<think>(.*?)$',
                         r'<details open style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理中...</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    if '<think>' not in content and '</think>' in content:
        content = re.sub(r'(.*?)</think>',
                         r'<details style="font-style: italic; background: rgba(222, 222, 222, 0.5); padding: 10px; border-radius: 10px;"><summary style="font-weight:bold;">推理内容（展开）</summary>\1</details>',
                         content,
                         flags=re.DOTALL)

    return content


config = {
    "out_dir": "out",
    "lora_name": "cooking1e-4",
    "num_hidden_layers": 8,
    "hidden_size": 512,
}


def init_model():
    tokenizer = AutoTokenizer.from_pretrained('../model/')

    ckp = f'../{config["out_dir"]}/full_sft_512.pth'

    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        use_moe=False
    ))

    model.load_state_dict(torch.load(ckp, map_location='cuda'), strict=True)

    if config["lora_name"] != 'None':
        apply_lora(model)
        load_lora(model, f'../{config["out_dir"]}/lora/{config["lora_name"]}_512.pth')

    return model.eval().to('cuda'), tokenizer


def clear_chat_messages():
    del st.session_state.messages
    del st.session_state.chat_messages


def init_chat_messages():
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar=image_url):
                    st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                    if st.button("🗑", key=f"delete_{i}"):
                        st.session_state.messages.pop(i)
                        st.session_state.messages.pop(i - 1)
                        st.session_state.chat_messages.pop(i)
                        st.session_state.chat_messages.pop(i - 1)
                        st.rerun()
            else:
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: #ddd; border-radius: 10px; color: black;">{message["content"]}</div></div>',
                    unsafe_allow_html=True)

    else:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    return st.session_state.messages

def regenerate_answer(index):
    st.session_state.messages.pop()
    st.session_state.chat_messages.pop()
    st.rerun()


def delete_conversation(index):
    st.session_state.messages.pop(index)
    st.session_state.messages.pop(index - 1)
    st.session_state.chat_messages.pop(index)
    st.session_state.chat_messages.pop(index - 1)
    st.rerun()


st.sidebar.title("模型设定调整")

st.session_state.history_chat_num = st.sidebar.slider("历史对话数", 0, 6, 0, step=2)
st.session_state.max_new_tokens = st.sidebar.slider("最大序列长度", 256, 8192, 8192, step=1)
st.session_state.temperature = st.sidebar.slider("模型温度", 0.6, 1.2, 0.85, step=0.01)

model_source = st.sidebar.radio("选择模型来源", ["本地模型"], index=0)

MODEL_PATHS = {
    "做饭糕手": ["../MiniMind2", "MiniMind2"],
}

selected_model = st.sidebar.selectbox('选择模型', list(MODEL_PATHS.keys()), index=0)  # 默认选择 MiniMind2
model_path = MODEL_PATHS[selected_model][0]
slogan = f"你好，我是杜健和崔凯乾开发的Milkmind龙芯版"

image_url = "../pictures/img.png"

st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
    '</div>'
    '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">内容完全由AI生成，吃出问题概不负责。'
    '</div>',
    unsafe_allow_html=True
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    if model_source == "本地模型":
        model, tokenizer = init_model()
    else:
        model, tokenizer = None, None

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    messages = st.session_state.messages

    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                if st.button("×", key=f"delete_{i}"):
                    st.session_state.messages = st.session_state.messages[:i - 1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i - 1]
                    st.rerun()
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{message["content"]}</div></div>',
                unsafe_allow_html=True)

    prompt = st.chat_input(key="input", placeholder="和Milkmind龙芯版对话：")

    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        regenerate_index = st.session_state.regenerate_index
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px;  background-color: gray; border-radius: 10px; color:white; ">{prompt}</div></div>',
            unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
        st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()

            random_seed = random.randint(0, 2 ** 32 - 1)
            setup_seed(random_seed)

            st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[
                                                             -(st.session_state.history_chat_num + 1):]
            new_prompt = tokenizer.apply_chat_template(
                st.session_state.chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                new_prompt,
                return_tensors="pt",
                truncation=True
            ).to(device)

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
                "num_return_sequences": 1,
                "do_sample": True,
                "attention_mask": inputs.attention_mask,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": st.session_state.temperature,
                "top_p": 0.85,
                "streamer": streamer,
            }

            Thread(target=model.generate, kwargs=generation_kwargs).start()

            answer = ""
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

            messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            with st.empty():
                if st.button("×", key=f"delete_{len(messages) - 1}"):
                    st.session_state.messages = st.session_state.messages[:-2]
                    st.session_state.chat_messages = st.session_state.chat_messages[:-2]
                    st.rerun()


if __name__ == "__main__":
    from transformers import AutoTokenizer, TextIteratorStreamer

    main()
