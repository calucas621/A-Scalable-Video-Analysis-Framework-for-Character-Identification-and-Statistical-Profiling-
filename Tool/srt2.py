# === 升級版：文字特徵工程 + 中文輸出 + 英文分析（已移除角色名檢測）===
import re
import os
import chardet
import csv
from transformers import pipeline


#字幕拿掉角色版本

# 關閉警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === 設定路徑 ===
OUTPUT_DIR = r"C:\Users\user\Desktop\output\12_腦筋急轉彎2\字幕特徵"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "feature_results.csv")

# === 解析 SRT ===
def parse_srt(file_path):
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案！\n {file_path}")
        return []
    with open(file_path, 'rb') as f:
        raw = f.read()
        encoding = chardet.detect(raw)['encoding']
        print(f"偵測到編碼: {encoding}")
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        content = f.read()
    subtitles = []
    blocks = re.split(r'\n\n+', content.strip())
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 3:
            index = lines[0]
            time = lines[1]
            text = ' '.join(lines[2:]).replace('\n', ' ')
            start_time = time.split(' --> ')[0]
            end_time = time.split(' --> ')[1]
            subtitles.append({
                'index': index,
                'start_time': start_time,
                'end_time': end_time,
                'text': text
            })
    return subtitles

# === 時間轉秒（已修復毫秒問題）===
def time_to_seconds(t):
    """
    支援格式：
    HH:MM:SS,mmm
    HH:MM:SS.mmm
    HH:MM:SS
    """
    match = re.match(r'(\d+):(\d+):(\d+)[,.]?(\d+)?', t)
    if not match:
        return 0.0

    h, m, s, ms = match.groups()
    ms = ms if ms else '0'
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# === 特徵計算函數（已移除 has_name_mention）===
def extract_features(text_en, text_zh, start_time, end_time):
    duration = time_to_seconds(end_time) - time_to_seconds(start_time)
    duration = max(duration, 0.1)  # 避免除以 0

    words = text_en.split()
    num_words = len(words)
    text_length = len(text_en)
    avg_word_length = sum(len(w) for w in words) / num_words if num_words else 0

    num_exclamations = text_en.count('!')
    num_questions = text_en.count('?')
    has_uppercase = int(any(c.isupper() and c.isalpha() for c in text_en))

    # 關鍵字特徵
    question_words = {'what', 'why', 'how', 'when', 'where', 'who'}
    swear_words = {'damn', 'hell', 'shit', 'fuck', 'bitch', 'ass'}
    action_words = {'run', 'jump', 'fight', 'hit', 'shoot', 'kill', 'grab', 'throw'}
    greetings = {'hi', 'hello', 'hey', 'yo'}
    thanks = {'thanks', 'thank', 'appreciate'}

    has_question_word = int(any(w.lower() in question_words for w in words))
    has_swear = int(any(w.lower() in swear_words for w in words))
    has_action_word = int(any(w.lower() in action_words for w in words))
    has_greeting = int(any(w.lower() in greetings for w in words))
    has_thank = int(any(w.lower() in thanks for w in words))

    # 關鍵字總數（已不包含角色名）
    keyword_count = sum([has_question_word, has_swear, has_action_word,
                         has_greeting, has_thank])

    # 標點與密度統計
    punctuation = sum(1 for c in text_en if c in '.,!?;:"()')
    punctuation_ratio = punctuation / text_length if text_length else 0
    capital_ratio = sum(1 for c in text_en if c.isupper()) / text_length if text_length else 0
    text_density = num_words / duration if duration > 0 else 0

    return {
        'start_time': start_time,
        'num_words': num_words,
        'avg_word_length': round(avg_word_length, 3),
        'text_length': text_length,
        'num_exclamations': num_exclamations,
        'num_questions': num_questions,
        'has_uppercase': has_uppercase,
        'has_question_word': has_question_word,
        'has_swear': has_swear,
        'has_action_word': has_action_word,
        'has_greeting': has_greeting,
        'has_thank': has_thank,
        'keyword_count': keyword_count,
        'text_density': round(text_density, 3),
        'punctuation_ratio': round(punctuation_ratio, 3),
        'capital_ratio': round(capital_ratio, 3),
        'line_index': 0,  # 後面填
        'subtitle_text': text_zh
    }

# === 主程式 ===
def analyze_features(file_path):
    subtitles = parse_srt(file_path)
    if not subtitles:
        return

    print("載入翻譯模型（中→英）...")
    try:
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
    except Exception as e:
        print("翻譯模型載入失敗 → 請執行：pip install sentencepiece transformers")
        print(f"錯誤訊息：{e}")
        return

    results = []
    print("\n正在分析特徵...\n")
    for i, sub in enumerate(subtitles):
        sub['line_index'] = i + 1
        text_zh = sub['text']

        # 翻譯成英文進行特徵分析
        try:
            text_en = translator(text_zh, max_length=512)[0]['translation_text']
        except:
            print(f"第 {i+1} 行翻譯失敗，使用原文")
            text_en = text_zh

        features = extract_features(text_en, text_zh, sub['start_time'], sub['end_time'])
        features['line_index'] = i + 1
        results.append(features)

        print(f"[{sub['start_time']}] {text_zh[:50]}{'...' if len(text_zh)>50 else ''}")

    # === 寫入 CSV（已移除 has_name_mention）===
    fieldnames = [
        'start_time', 'line_index', 'num_words', 'avg_word_length', 'text_length',
        'num_exclamations', 'num_questions', 'has_uppercase',
        'has_question_word', 'has_swear', 'has_action_word', 'has_greeting',
        'has_thank', 'keyword_count',
        'text_density', 'punctuation_ratio', 'capital_ratio', 'subtitle_text'
    ]

    with open(CSV_OUTPUT, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n特徵分析完成！（已移除角色名檢測）")
    print(f"CSV 已儲存：{CSV_OUTPUT}")

# === 執行 ===
if __name__ == "__main__":
    srt_file_path = r"C:\Users\user\Desktop\output\12_腦筋急轉彎2\Inside Out 2 Features\Inside Out 2 字幕.srt"

    if not os.path.exists(srt_file_path):
        print(f"錯誤：找不到字幕檔案\n{srt_file_path}")
    else:
        analyze_features(srt_file_path)
