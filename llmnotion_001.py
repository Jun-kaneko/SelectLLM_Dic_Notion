import os
import re
from typing import List, Tuple, Optional, Generator
from openai import OpenAI
import anthropic
import gradio as gr
import gradio.themes.base as gr_themes_base
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    # notion接続用の設定
    from notion_client import Client
    notion_client_instance = Client(auth=os.getenv("NOTION_API_KEY"))
    NOTION_DATABASE_ID = os.getenv('NOTION_DATABASE_ID')
    NOTION_DATABASE_ID_WORDS = os.getenv('NOTION_DATABASE_ID_WORDS')
    if not NOTION_DATABASE_ID:
        print("WARNING: 環境変数 'NOTION_DATABASE_ID' が設定されていません。Notion連携は機能しません。")
    if not NOTION_DATABASE_ID_WORDS:
        print("WARNING: 環境変数 'NOTION_DATABASE_ID_WORDS' が設定されていません。Notion連携は機能しません。")
except ImportError:
    notion_client_instance = None
    NOTION_DATABASE_ID = None
    NOTION_DATABASE_ID_WORDS = None
    print("WARNING: 'notion-client' ライブラリが見つかりません。Notion連携は機能しません。'pip install notion-client' でインストールしてください。")
except Exception as e:
    notion_client_instance = None
    NOTION_DATABASE_ID = None
    NOTION_DATABASE_ID_WORDS = None
    print(f"WARNING: Notionクライアントの初期化に失敗しました: {e}。Notion連携は機能しません。")

# APIキーの読み込み (環境変数から)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

# 各モデルのクライアント初期化
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
claude_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None

geminiclient = OpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
) if google_api_key else None

deepseek_via_openai_client = OpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
) if deepseek_api_key else None

# --- システムプロンプトの定義 ---
DEFAULT_SYSTEM_PROMPT = "あなたはフレンドリーな優秀なアシスタントです。質問に対してわかりやすく回答をしてください。回答は日本語です。"
ENGLISH_WORD_SYSTEM_PROMPT = "あなたはフレンドリーな優秀な英語の講師です。英語の文法、単語について聞かれたら、英語学習者中級レベルの人に対してわかりやすく説明をしてください。また、例や面白情報があると良いです。回答は日本語です。"

# 各モデルからのストリーミング応答を生成する関数群
def stream_gpt(messages: List[dict]):
    if not openai_client:
        yield "Error: OpenAI API Key is not set."
        return
    try:
        stream = openai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            stream=True
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
    except Exception as e:
        yield f"Error during GPT streaming: {e}"

def stream_gemini(messages: List[dict]):
    if not geminiclient:
        yield "Error: Google API Key is not set."
        return
    try:
        stream = geminiclient.chat.completions.create(
            model='gemini-2.5-flash',
            messages=messages,
            stream=True
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
    except Exception as e:
        yield f"Error during Gemini streaming: {e}"

def stream_deepseek(messages: List[dict]):
    if not deepseek_via_openai_client:
        yield "Error: DeepSeek API Key is not set."
        return
    try:
        stream = deepseek_via_openai_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
    except Exception as e:
        yield f"Error during DeepSeek streaming: {e}"

def stream_claude(system_prompt: str, messages_for_claude: List[dict]):
    if not claude_client:
        yield "Error: Anthropic API Key is not set."
        return
    try:
        result = claude_client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0.7,
            system=system_prompt,
            messages=messages_for_claude
        )
        with result as stream:
            for text in stream.text_stream:
                yield text or ""
    except Exception as e:
        yield f"Error during Claude streaming: {e}"

# ヘルパー関数: 内部履歴 (List[Dict]) を Gradio Chatbot 表示形式 (List[List[str]]) に変換
# chat_stateは [{"role": "user", "content": "..." }, {"role": "assistant", "content": "..." }, ...] の形式を想定
def _convert_chat_state_to_chatbot_display(chat_state: List[dict]) -> List[List[Optional[str]]]:
    chatbot_display = []
    for i in range(0, len(chat_state), 2):
        user_msg = chat_state[i]["content"] if chat_state[i]["role"] == "user" else None
        assistant_msg = chat_state[i+1]["content"] if i+1 < len(chat_state) and chat_state[i+1]["role"] == "assistant" else None
        if user_msg is not None:
            chatbot_display.append([user_msg, assistant_msg])
    return chatbot_display

# モデルからの応答をストリーミングし、チャット履歴を更新するメイン関数
# system_prompt_textbox_value と question_type を受け取る
def stream_model_with_history(
    system_prompt_textbox_value: str, # 「通常の質問」時に使用するシステムプロンプト (テキストボックスの値)
    question_type: str,               # ユーザーが選択した質問タイプ ("通常の質問" or "英単語の質問")
    user_input: str,
    model: str,
    chat_state: List[dict] # 内部履歴 (Dict形式)
) -> Generator[Tuple[List[List[Optional[str]]], List[dict]], None, None]: # (Gradio Chatbot表示, 更新された内部履歴)

    # 質問タイプに基づいて、実際にモデルに渡すシステムプロンプトを決定
    actual_system_prompt = ""
    if question_type == "通常の質問":
        actual_system_prompt = system_prompt_textbox_value # テキストボックスの値をそのまま使用
    elif question_type == "英単語の質問":
        actual_system_prompt = ENGLISH_WORD_SYSTEM_PROMPT # 定義済みの英語講師プロンプトを使用
    else:
        # 予期しない質問タイプの場合のフォールバック (念のため)
        actual_system_prompt = DEFAULT_SYSTEM_PROMPT


    # 1. 内部履歴 (chat_state) を Gradio Chatbot 表示形式に変換
    chatbot_display = _convert_chat_state_to_chatbot_display(chat_state)

    # 2. 新しいユーザー入力を Chatbot 表示に追加 (アシスタント応答はまだ空)
    chatbot_display.append([user_input, None])

    # 3. モデルへのメッセージを組み立て
    base_messages_for_model = chat_state[:]
    # OpenAIスタイルのAPI (GPT, Gemini, DeepSeek) 用のメッセージ形式
    openai_style_messages = [{"role": "system", "content": actual_system_prompt}] + base_messages_for_model + [{"role": "user", "content": user_input}]
    # Claude API 用のメッセージ形式 (systemプロンプトは別途引数で渡す)
    claude_style_messages = base_messages_for_model + [{"role": "user", "content": user_input}]

    gen = None
    if model=="GPT":
        gen = stream_gpt(openai_style_messages)
    elif model=="Gemini":
        gen = stream_gemini(openai_style_messages)
    elif model=="DeepSeek":
        gen = stream_deepseek(openai_style_messages)
    elif model=="Claude":
        gen = stream_claude(actual_system_prompt, claude_style_messages)
    else:
        error_msg = f"Error: 選択されたモデル '{model}' はサポートされていません。"
        chatbot_display[-1][1] = error_msg
        yield chatbot_display, chat_state
        return

    if gen is None:
        error_msg = "Error: モデルのジェネレータを初期化できませんでした (APIキーの不足など)。"
        chatbot_display[-1][1] = error_msg
        yield chatbot_display, chat_state
        return

    # 4. 逐次テキストを累積して返す（Gradio ストリーミング）
    acc = ""
    for piece in gen:
        acc += piece
        chatbot_display[-1][1] = acc
        yield chatbot_display, chat_state

    # 5. 応答完了後に内部履歴 (chat_state) を更新
    new_chat_state = chat_state + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": acc},
    ]
    yield chatbot_display, new_chat_state

# チャット履歴をクリアする関数
def clear_chat_history():
    return [], []

# --- Notion連携関数 ---
# 2000文字を超えるテキストをNotionのparagraphブロックに分割するヘルパー関数
def split_text_into_blocks(text: str, max_length: int = 1900) -> List[dict]:
    """
    長いテキストをNotionのparagraphブロックの文字数制限に合わせて分割
    """
    blocks = []
    current_pos = 0
    while current_pos < len(text):
        # 残りのテキストが max_length より短ければそのまま追加
        if len(text) - current_pos <= max_length:
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": text[current_pos:]}}]}
            })
            current_pos = len(text)
        else:
            # max_lengthまでで分割
            # シンプルに max_length で区切る
            chunk = text[current_pos : current_pos + max_length]
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}
            })
            current_pos += max_length
    return blocks


def send_to_notion(chat_history: List[dict], question_type) -> str:
    """チャット履歴をNotionデータベースに送信 """
    if not notion_client_instance or not NOTION_DATABASE_ID or not NOTION_DATABASE_ID_WORDS:
        return "Notion連携が設定されていません。環境変数 NOTION_API_KEY と NOTION_DATABASE_ID と NOTION_DATABSE_ID_WORDS を確認してください。"

    if not chat_history:
        return "Notionに送信するチャット履歴がありません。"

    try:
        # ページのタイトルを生成 (最初のユーザーメッセージと日時を使用)
        first_user_message = next((item['content'] for item in chat_history if item['role'] == 'user'), "Untitled Chat Log")
        if question_type == "通常の質問":
            page_title = f"{first_user_message[:50]}{'...' if len(first_user_message) > 50 else ''} " \
             f"({datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M')})"
        elif question_type == "英単語の質問":
            # 最初の英語の単語を探す
            match = re.search(r"[A-Za-z]+", first_user_message)
            first_word = match.group() if match else "NoTitle"
            page_title = f"{first_word} ({datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M')})"

        # Notionのブロックとしてチャット履歴を整形
        children_blocks = []
        for item in chat_history:
            prefix = ""
            content = item['content']
            if item['role'] == 'user':
                prefix = "🧑 User: "
            elif item['role'] == 'assistant':
                prefix = "🤖 Assistant: "

            # プレフィックスとコンテンツを結合
            full_text_to_send = prefix + content

            # テキストを分割してブロックに追加
            # プレフィックスも含めて2000文字制限に収まるように分割する
            max_block_content_length = 1900 # Notionの文字数制限。2000文字に余裕を持たせるので1900で定義

            # 初回ブロックにはプレフィックスを付ける
            first_chunk_text = full_text_to_send[0:max_block_content_length]
            children_blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": first_chunk_text}}]}
            })

            # 1900文字を超える場合は、残りを追加のブロックとして分割
            remaining_text = full_text_to_send[max_block_content_length:]
            current_pos = 0
            while current_pos < len(remaining_text):
                chunk = remaining_text[current_pos : current_pos + max_block_content_length]
                children_blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}
                })
                current_pos += max_block_content_length

        # Notionに新しいページを作成
        if question_type == "通常の質問":
            database_id = NOTION_DATABASE_ID
        elif question_type == "英単語の質問":
            database_id = NOTION_DATABASE_ID_WORDS
        notion_client_instance.pages.create(
            parent={"database_id": database_id},
            properties={
                "Name": { # Notionデータベースのタイトルプロパティ名に合わせる（通常は"Name"）
                    "title": [
                        {
                            "text": {
                                "content": page_title
                            }
                        }
                    ]
                }
            },
            children=children_blocks
        )
        return "Notionにチャット履歴を送信しました！✨"
    except Exception as e:
        print(f"Notionへの送信中にエラーが発生しました: {e}")
        return f"Notionへの送信に失敗しました: {e}"

# Gradioインターフェースの構築
def build_interface():
    with gr.Blocks(title="LLM Chat App", theme=gr_themes_base.Base(primary_hue="blue", secondary_hue="emerald")) as communication:
        gr.Markdown("# LLM Chat App (GPT, Claude, Gemini, DeepSeek)")

        # システムプロンプトとモデル選択を横に並べる
        with gr.Row():
            # 質問タイプ選択
            question_type_radio = gr.Radio(
                ["通常の質問", "英単語の質問"],
                label="質問タイプを選択",
                value="通常の質問", # 初期値は「通常の質問」
                scale=1
            )
            # 「通常の質問」用のシステムプロンプト (テキストボックス)
            # 「英単語の質問」選択時にはこの値は無視
            system_prompt_textbox = gr.Textbox(
                label="通常の質問用システムプロンプト(英単語質問の場合、内部的にシステムプロンプトが切り替わりますが、ここの表示は変わりません)",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=3,
                scale=2,
                interactive=True
            )
            model_selector = gr.Dropdown(
                ["GPT", "Claude", "Gemini", "DeepSeek"],
                label="Select model",
                value="Gemini",
                scale=1
            )

        # チャット履歴を保持するState (内部的なDict形式の履歴)
        chat_history = gr.State([])
        # Gradio Chatbot コンポーネント (表示用)
        chatbot = gr.Chatbot(
            label="チャット履歴",
            height=400,
            avatar_images=(None, "./images/bluebird_robot_bot.png"),
        )

        # ユーザー入力欄
        user_input = gr.Textbox(label="ユーザ入力", lines=6, placeholder="ここにメッセージを入力してください...")

        # ボタン類を横に並べる
        with gr.Row():
            submit_button = gr.Button("送信", variant="primary")
            clear_history_button = gr.Button("クリア", variant="secondary")
            # Notion送信ボタン
            notion_send_button = gr.Button("Notionに送信する", variant="secondary")
        # Notion送信結果のメッセージを表示する場所
        notion_status_message = gr.Markdown(value="", visible=False)

        # イベントハンドラの定義
        # 送信ボタンがクリックされたとき
        submit_event = submit_button.click(
            fn=stream_model_with_history,
            inputs=[system_prompt_textbox, question_type_radio, user_input, model_selector, chat_history],
            outputs=[chatbot, chat_history]
        )
        # ストリーミング完了後、ユーザー入力欄をクリア
        submit_event.then(
            lambda: gr.update(value=""),
            inputs=None,
            outputs=[user_input]
        )

        # ユーザー入力欄でEnterが押されたとき
        user_input.submit(
            fn=stream_model_with_history,
            inputs=[system_prompt_textbox, question_type_radio, user_input, model_selector, chat_history],
            outputs=[chatbot, chat_history]
        ).then(
            lambda: gr.update(value=""),
            inputs=None,
            outputs=[user_input]
        )

        # 「クリア」ボタンがクリックされたとき
        clear_history_button.click(
            fn=clear_chat_history,
            outputs=[chat_history, chatbot]
        ).then(
            lambda: gr.update(value=""), # ユーザー入力欄もクリア
            inputs=None,
            outputs=[user_input]
        )

        # Notion送信ボタンがクリックされたとき
        notion_send_button.click(
            fn=send_to_notion,
            inputs=[chat_history, question_type_radio],
            outputs=[notion_status_message] # 結果メッセージを表示
        ).then(
            lambda: gr.update(visible=True), # メッセージを可視化
            inputs=None,
            outputs=[notion_status_message]
        )

    return communication

if __name__ == "__main__":
    app = build_interface()
    port = int(os.getenv("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)