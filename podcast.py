import os  
import tempfile  
import time  
import json  
import datetime  
import numpy as np  
import librosa  
import soundfile as sf  
import speech_recognition as sr  
from pydub import AudioSegment, silence, effects  
import matplotlib.pyplot as plt  
import subprocess  
import shutil  
from pathlib import Path  
import base64  
import streamlit as st  


def initialize_session_state():  
    """Streamlitのセッション状態を初期化する関数"""  
    if 'audio_data' not in st.session_state:  
        st.session_state.audio_data = None  
    if 'waveform' not in st.session_state:  
        st.session_state.waveform = None  
    if 'sample_rate' not in st.session_state:  
        st.session_state.sample_rate = None  
    if 'segments' not in st.session_state:  
        st.session_state.segments = []  
    if 'transcript' not in st.session_state:  
        st.session_state.transcript = ""  
    if 'processed_audio' not in st.session_state:  
        st.session_state.processed_audio = None  
    if 'temp_dir' not in st.session_state:  
        st.session_state.temp_dir = tempfile.mkdtemp()  
    if 'original_audio_format' not in st.session_state:  
        st.session_state.original_audio_format = None
    if 'preset_preview' not in st.session_state:
        st.session_state.preset_preview = None
    if 'bgm_files' not in st.session_state:
        st.session_state.bgm_files = {}
  
initialize_session_state()  


st.set_page_config(page_title="ポッドキャスト自動音声編集アプリ", layout="wide")  
st.title("ポッドキャスト自動音声編集アプリ")  
  
def get_binary_file_downloader_html(bin_file, file_label='File'):  
    """  
    バイナリファイルをダウンロードできるHTMLリンクを生成する関数  
      
    Args:  
        bin_file (str): ダウンロードするファイルのパス  
        file_label (str): リンクに表示するラベル  
      
    Returns:  
        str: ダウンロードリンクのHTML  
    """  
    with open(bin_file, 'rb') as f:  
        data = f.read()  
    bin_str = base64.b64encode(data).decode()  
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'  
    return href  
  
# タブの数を減らして3つに変更
tab1, tab2, tab3 = st.tabs(["編集", "プレビュー", "エクスポート"])  

# サイドバーにプリセット管理の領域を追加
with st.sidebar:
    st.header("ファイルのアップロード")
    uploaded_file = st.file_uploader("音声ファイルをアップロード", type=['wav', 'mp3', 'ogg', 'flac'])
    
    st.header("プリセット選択")

    # プリセットの取得・保存関数
    def get_presets():
        preset_dir = os.path.join(os.path.expanduser("~"), "podcast_editor_presets")
        os.makedirs(preset_dir, exist_ok=True)
        preset_file = os.path.join(preset_dir, "presets.json")
        if os.path.exists(preset_file):
            with open(preset_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # デフォルトプリセットを作成して保存
            default_presets = {
                "標準": {
                    "noise_reduction": 0.01,
                    "silence_threshold": -40,
                    "min_silence_duration": 300,
                    "volume_normalize": True,
                    "intro_music": "なし",
                    "add_transitions": False,
                    "language": "日本語",
                    "bgm_file": "",
                    "bgm_volume": -20
                },
                "ラジオ風": {
                    "noise_reduction": 0.2,
                    "silence_threshold": -45,
                    "min_silence_duration": 800,
                    "volume_normalize": True,
                    "intro_music": "プロフェッショナル",
                    "add_transitions": True,
                    "language": "日本語",
                    "bgm_file": "",
                    "bgm_volume": -25
                },
                "インタビュー": {
                    "noise_reduction": 0.15,
                    "silence_threshold": -35,
                    "min_silence_duration": 1200,
                    "volume_normalize": True,
                    "intro_music": "なし",
                    "add_transitions": False,
                    "language": "日本語",
                    "bgm_file": "",
                    "bgm_volume": -30
                }
            }
            with open(preset_file, "w", encoding="utf-8") as f:
                json.dump(default_presets, f, ensure_ascii=False, indent=2)
            return default_presets
    
    def save_presets(presets):
        preset_dir = os.path.join(os.path.expanduser("~"), "podcast_editor_presets")
        os.makedirs(preset_dir, exist_ok=True)
        preset_file = os.path.join(preset_dir, "presets.json")
        with open(preset_file, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
    
    # BGMファイルの管理
    def get_bgm_files():
        bgm_dir = os.path.join(os.path.expanduser("~"), "podcast_editor_bgm")
        os.makedirs(bgm_dir, exist_ok=True)
        bgm_list = {}
        for file in os.listdir(bgm_dir):
            if file.lower().endswith(('.mp3', '.wav', '.ogg')):
                bgm_list[file] = os.path.join(bgm_dir, file)
        return bgm_list
    
    # BGMファイルを取得
    st.session_state.bgm_files = get_bgm_files()
    
    # プリセットを取得
    presets = get_presets()
    selected_preset = st.selectbox("プリセット", list(presets.keys()))
    
    # プリセット管理のエキスパンダー
    with st.expander("プリセット管理"):
        preset_action = st.radio("アクション", ["新規作成", "編集", "削除"], horizontal=True)
        
        if preset_action == "新規作成":
            new_preset_name = st.text_input("新しいプリセット名")
            if st.button("作成") and new_preset_name and new_preset_name not in presets:
                presets[new_preset_name] = {
                    "noise_reduction": 0.1,
                    "silence_threshold": -40,
                    "min_silence_duration": 1000,
                    "volume_normalize": True,
                    "intro_music": "なし",
                    "add_transitions": False,
                    "language": "日本語",
                    "bgm_file": "",
                    "bgm_volume": -20
                }
                save_presets(presets)
                st.success(f"プリセット '{new_preset_name}' を作成しました")
                st.experimental_rerun()
        
        elif preset_action == "編集":
            edit_preset = st.selectbox("編集するプリセット", list(presets.keys()), key="edit_preset")
            
            # 現在のプリセット値を取得
            current_preset = presets[edit_preset]
            
            # 各パラメータの編集UI
            noise_reduction = st.slider("ノイズリダクションレベル", 0.0, 1.0, current_preset["noise_reduction"], 0.01)
            silence_threshold = st.slider("沈黙検出閾値(dB)", -60, -20, current_preset["silence_threshold"], 1)
            min_silence_duration = st.slider("最小沈黙時間(ms)", 300, 2000, current_preset["min_silence_duration"], 50)
            volume_normalize = st.checkbox("音量ノーマライゼーション", current_preset["volume_normalize"])
            intro_music = st.selectbox("イントロ音楽", ["なし", "プロフェッショナル", "アップビート", "リラックス"], 
                                    ["なし", "プロフェッショナル", "アップビート", "リラックス"].index(current_preset["intro_music"]))
            add_transitions = st.checkbox("セグメント間にトランジションを追加", current_preset["add_transitions"])
            language = st.selectbox("音声認識言語", ["日本語", "英語", "スペイン語"], 
                                ["日本語", "英語", "スペイン語"].index(current_preset["language"]))
            
            # BGM設定
            bgm_options = ["なし"] + list(st.session_state.bgm_files.keys())
            bgm_file = st.selectbox("BGM選択", bgm_options)
            bgm_volume = st.slider("BGM音量 (dB)", -40, 0, current_preset.get("bgm_volume", -20), 1)
            
            # プレビュー機能
            if st.button("プリセットをプレビュー") and st.session_state.audio_data:
                with st.spinner('プレビュー生成中...'):
                    # プレビュー用の処理を実行
                    y_reduced = reduce_noise(st.session_state.waveform, st.session_state.sample_rate, noise_reduction)
                    preview_path = os.path.join(st.session_state.temp_dir, "preview.wav")
                    sf.write(preview_path, y_reduced, st.session_state.sample_rate)
                    processed_audio = AudioSegment.from_file(preview_path)
                    
                    if volume_normalize:
                        processed_audio = normalize_audio(processed_audio)
                    
                    segments = segment_audio(processed_audio, silence_threshold, min_silence_duration)
                    
                    final_audio = add_sound_effects(processed_audio, intro_music, add_transitions, segments)
                    
                    # BGM追加
                    if bgm_file != "なし":
                        try:
                            bgm_path = st.session_state.bgm_files[bgm_file]
                            bgm_audio = AudioSegment.from_file(bgm_path)
                            
                            # BGMの長さを調整
                            if len(bgm_audio) < len(final_audio):
                                # 必要な回数だけBGMを繰り返す
                                repeats = int(len(final_audio) / len(bgm_audio)) + 1
                                bgm_extended = bgm_audio * repeats
                                bgm_extended = bgm_extended[:len(final_audio)]
                            else:
                                bgm_extended = bgm_audio[:len(final_audio)]
                            
                            # BGMの音量を調整
                            bgm_extended = bgm_extended.apply_gain(bgm_volume)
                            
                            # BGMとボイスをミックス
                            final_audio = final_audio.overlay(bgm_extended)
                        except Exception as e:
                            st.error(f"BGM処理エラー: {e}")
                    
                    preview_output = os.path.join(st.session_state.temp_dir, "preset_preview.wav")
                    final_audio.export(preview_output, format="wav")
                    st.session_state.preset_preview = preview_output
            
            # プレビュー再生
            if st.session_state.preset_preview:
                st.audio(st.session_state.preset_preview)
            
            # 保存ボタン
            if st.button("プリセットを保存"):
                presets[edit_preset] = {
                    "noise_reduction": noise_reduction,
                    "silence_threshold": silence_threshold,
                    "min_silence_duration": min_silence_duration,
                    "volume_normalize": volume_normalize,
                    "intro_music": intro_music,
                    "add_transitions": add_transitions,
                    "language": language,
                    "bgm_file": bgm_file if bgm_file != "なし" else "",
                    "bgm_volume": bgm_volume
                }
                save_presets(presets)
                st.success(f"プリセット '{edit_preset}' を更新しました")
                st.experimental_rerun()
        
        elif preset_action == "削除":
            delete_preset = st.selectbox("削除するプリセット", list(presets.keys()), key="delete_preset")
            if st.button("削除") and delete_preset in presets and len(presets) > 1:
                del presets[delete_preset]
                save_presets(presets)
                st.success(f"プリセット '{delete_preset}' を削除しました")
                st.experimental_rerun()
            elif st.button("削除") and len(presets) <= 1:
                st.error("最後のプリセットは削除できません")
    
    # BGM管理のエキスパンダー
    with st.expander("BGM管理"):
        uploaded_bgm = st.file_uploader("BGMファイルをアップロード", type=['wav', 'mp3', 'ogg'])
        if uploaded_bgm:
            bgm_dir = os.path.join(os.path.expanduser("~"), "podcast_editor_bgm")
            os.makedirs(bgm_dir, exist_ok=True)
            bgm_path = os.path.join(bgm_dir, uploaded_bgm.name)
            with open(bgm_path, "wb") as f:
                f.write(uploaded_bgm.getbuffer())
            st.success(f"BGM '{uploaded_bgm.name}' を追加しました")
            st.session_state.bgm_files = get_bgm_files()
            st.experimental_rerun()
        
        # 既存BGMの一覧と削除
        st.write("登録済みBGM:")
        for bgm_name in st.session_state.bgm_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(bgm_name)
            with col2:
                if st.button("削除", key=f"del_{bgm_name}"):
                    os.remove(st.session_state.bgm_files[bgm_name])
                    st.success(f"BGM '{bgm_name}' を削除しました")
                    st.session_state.bgm_files = get_bgm_files()
                    st.experimental_rerun()

# 選択したプリセットの設定を取得する関数
def get_preset_settings(preset_name):
    preset_dir = os.path.join(os.path.expanduser("~"), "podcast_editor_presets")
    preset_file = os.path.join(preset_dir, "presets.json")
    if os.path.exists(preset_file):
        with open(preset_file, "r", encoding="utf-8") as f:
            presets = json.load(f)
        if preset_name in presets:
            return presets[preset_name]
    return None

def load_audio(file):  
    """  
    音声ファイルを開いてセッション状態に保存する関数  
      
    Args:  
        file (BytesIO): ストリームリットのファイルアップローダーからの音声ファイル  
      
    Returns:  
        tuple: 音声データ（numpy array）、サンプリングレート、Waveファイルのパス、AudioSegmentオブジェクト  
    """  
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:  
        tmp_file.write(file.getvalue())  
        tmp_path = tmp_file.name  
        audio_segment = AudioSegment.from_file(tmp_path)  
        wav_path = os.path.join(st.session_state.temp_dir, "original.wav")  
        audio_segment.export(wav_path, format="wav")  
        y, sample_rate = librosa.load(wav_path, sr=None)  
        st.session_state.original_audio_format = file.name.split('.')[-1]  
        return y, sample_rate, wav_path, audio_segment  
  
def plot_waveform(y, sample_rate):  
    """  
    波形をプロットする関数  
      
    Args:  
        y (numpy array): 音声データ  
        sample_rate (int): サンプリングレート  
      
    Returns:  
        Figure: MatplotlibのFigureオブジェクト  
    """  
    fig, ax = plt.subplots(figsize=(10, 2))  
    librosa.display.waveshow(y, sr=sample_rate, ax=ax)  
    ax.set_title('音声波形')  
    ax.set_xlabel('時間 (秒)')  
    ax.set_ylabel('振幅')  
    return fig  

def plot_speaker_identification(waveform, sample_rate, speaker_segments):  
    """  
    話者識別結果を波形上に表示する関数  
      
    Args:  
        waveform (numpy array): 音声波形データ  
        sample_rate (int): サンプリングレート  
        speaker_segments (list): 話者識別結果のセグメントリスト  
      
    Returns:  
        Figure: MatplotlibのFigureオブジェクト  
    """  
    fig, ax = plt.subplots(figsize=(10, 3))  
    colors = {"話者A": "blue", "話者B": "red"}  
    for segment in speaker_segments:  
        start = segment["start"]  
        end = segment["end"]  
        speaker = segment["speaker"]  
        ax.axvspan(start, end, alpha=0.3, color=colors[speaker])  
    librosa.display.waveshow(waveform, sr=sample_rate, ax=ax)  
    ax.set_title("話者識別結果")  
    ax.legend(colors.keys())  
    return fig  
  
@st.cache_data  
def transcribe_audio(audio_path, language_code):  
    """  
    音声ファイルをテキストに書き起こす関数  
      
    Args:  
        audio_path (str): 音声ファイルのパス  
        language_code (str): 言語コード（例: "ja-JP"）  
      
    Returns:  
        str: 書き起こしたテキスト  
    """  
    try:  
        recognizer = sr.Recognizer()  
        audio_file = sr.AudioFile(audio_path)  
        with audio_file as source:  
            audio_data = recognizer.record(source)  
            text = recognizer.recognize_google(audio_data, language=language_code)  
        return text  
    except Exception as e:  
        return f"文字起こしエラー: {str(e)}"  
  
def segment_audio(audio_segment, silence_thresh, min_silence_len):  
    """  
    音声データを沈黙部分でセグメント化する関数  
      
    Args:  
        audio_segment (AudioSegment): 元の音声データ  
        silence_thresh (int): 沈黙とみなす音量閾値（dB）  
        min_silence_len (int): 最小沈黙時間（ms）  
      
    Returns:  
        list: セグメントのスタートとエンドの時間のリスト  
    """  
    silence_parts = silence.detect_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)  
    segments = []  
    prev_end = 0  
    for start, end in silence_parts:  
        if start > prev_end:  
            segments.append((prev_end, start))  
        prev_end = end  
    if prev_end < len(audio_segment):  
        segments.append((prev_end, len(audio_segment)))  
    return segments  

def reduce_noise(y, sample_rate, reduction_amount):  
    """  
    ノイズリダクションを行う関数  
      
    Args:  
        y (numpy array): 音声データ  
        sample_rate (int): サンプリングレート  
        reduction_amount (float): ノイズリダクションの強さ（0.0〜1.0）  
      
    Returns:  
        numpy array: ノイズリダクション後の音声データ  
    """  
    y_reduced = y.copy()  
    noise_mask = np.abs(y) < (reduction_amount * np.max(np.abs(y)))  
    y_reduced[noise_mask] = 0  
    return y_reduced  
  
def normalize_audio(audio_segment):  
    """  
    音量をノーマライゼーションする関数  
      
    Args:  
        audio_segment (AudioSegment): オーディオセグメント  
      
    Returns:  
        AudioSegment: ノーマライゼーション後のオーディオセグメント  
    """  
    return effects.normalize(audio_segment)  
  
def add_sound_effects(audio_segment, intro_type, add_transitions, segments):  
    """  
    効果音を追加する関数  
      
    Args:  
        audio_segment (AudioSegment): 元のオーディオセグメント  
        intro_type (str): イントロ音楽のタイプ（"なし", "プロフェッショナル", "アップビート", "リラックス"）  
        add_transitions (bool): セグメント間にトランジションを追加するかどうか  
        segments (list): オーディオセグメントのリスト  
      
    Returns:  
        AudioSegment: 効果音追加後のオーディオセグメント  
    """  
    result = AudioSegment.empty()  
    if intro_type != "なし":  
        intro = AudioSegment.silent(duration=3000)  
        intro = intro.fade_in(500).fade_out(500)  
        result += intro  
        result += AudioSegment.silent(duration=500)  
    for i, (start, end) in enumerate(segments):  
        segment = audio_segment[start:end]  
        result += segment  
        if add_transitions and i < len(segments) - 1:  
            segment = segment.fade_out(300)  
            transition = AudioSegment.silent(duration=300)  
            result += transition  
    return result  

def create_mp4(audio_path, output_path):  
    """  
    音声をMP4ファイルに変換する関数  
      
    Args:  
        audio_path (str): 音声ファイルのパス  
        output_path (str): 出力MP4ファイルのパス  
      
    Returns:  
        str: 出力MP4ファイルのパス  
    """  
    image_path = os.path.join(st.session_state.temp_dir, "placeholder.png")  
    plt.figure(figsize=(16, 9))  
    plt.text(0.5, 0.5, 'ポッドキャスト', horizontalalignment='center', verticalalignment='center', fontsize=40)  
    plt.axis('off')  
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)  
    plt.close()  
    cmd = [  
        'ffmpeg', '-y',  
        '-loop', '1',  
        '-i', image_path,  
        '-i', audio_path,  
        '-c:v', 'libx264',  
        '-tune', 'stillimage',  
        '-c:a', 'aac',  
        '-b:a', '192k',  
        '-pix_fmt', 'yuv420p',  
        '-shortest',  
        output_path  
    ]  
    subprocess.run(cmd, check=True)  
    return output_path  
  
def cut_audio_by_transcript(transcript, segments, audio_segment, keywords, padding_seconds=0.1):
    """  
    キーワードに基づいて音声をカットする関数  
      
    Args:  
        transcript (str): 音声の文字起こし結果  
        segments (list): オーディオセグメントのリスト  
        audio_segment (AudioSegment): オーディオセグメント  
        keywords (list): カットするキーワードのリスト  
        padding_seconds (float): キーワードの前後に追加でカットする時間（秒）
      
    Returns:  
        AudioSegment: キーワードでカット後のオーディオセグメント  
    """  
    result_audio = audio_segment
    cut_count = 0
    
    for keyword in keywords:
        if not keyword.strip():
            continue
            
        segments_to_cut = get_keyword_timestamps(transcript, segments, keyword)
        
        # 前後のパディングを追加してカット範囲を拡大
        padded_segments = []
        for start, end in segments_to_cut:
            padded_start = max(0, start - padding_seconds)
            padded_end = min(len(result_audio) / 1000, end + padding_seconds)
            padded_segments.append((padded_start, padded_end))
        
        # 重複を解消してマージ
        padded_segments.sort()
        merged_segments = []
        for segment in padded_segments:
            if not merged_segments or segment[0] > merged_segments[-1][1]:
                merged_segments.append(segment)
            else:
                merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], segment[1]))
        
        # 後ろから順にカット
        for start, end in reversed(merged_segments):
            result_audio = result_audio[:int(start * 1000)] + result_audio[int(end * 1000):]
            cut_count += 1
            
    return result_audio
  
def get_keyword_timestamps(transcript, segments, keyword):
    """  
    キーワードが見つかる時間（秒）範囲を返す関数  
      
    Args:  
        transcript (str): 音声の文字起こし結果  
        segments (list): オーディオセグメントのリスト  
        keyword (str): キーワード  
      
    Returns:  
        list: キーワードが見つかる開始時間と終了時間のリスト  
    """  
    timestamps = []
    
    # 文字起こしの中でキーワードを検索
    lower_transcript = transcript.lower()
    lower_keyword = keyword.lower()
    
    # 文字起こしの中のキーワードの位置を全て見つける
    positions = []
    start_pos = 0
    while True:
        pos = lower_transcript.find(lower_keyword, start_pos)
        if pos == -1:
            break
        positions.append((pos, pos + len(keyword)))
        start_pos = pos + 1
    
    # 各位置に対応する時間を計算
    total_duration = sum((end - start) for start, end in segments) / 1000  # 秒単位
    char_duration = total_duration / len(transcript) if transcript else 0
    
    for start_char, end_char in positions:
        start_time = start_char * char_duration
        end_time = end_char * char_duration
        timestamps.append((start_time, end_time))
    
    return timestamps

def transcribe_audio_partial(audio_segment, language_code, start, end, sample_rate):  
    """  
    サブセグメントの文字起こしを行う部分関数  
      
    Args:  
        audio_segment (AudioSegment): オーディオセグメント  
        language_code (str): 言語コード（例: "ja-JP"）  
        start (int): サブセグメントの開始位置  
        end (int): サブセグメントの終了位置  
        sample_rate (int): サンプリングレート  
      
    Returns:  
        str: サブセグメントの文字起こし結果  
    """  
    partial_audio = audio_segment[start:end]  
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{st.session_state.original_audio_format}') as tmp_file:  
        partial_audio.export(tmp_file.name, format=st.session_state.original_audio_format)  
        recognizer = sr.Recognizer()  
        audio_file = sr.AudioFile(tmp_file.name)  
        with audio_file as source:  
            audio_data = recognizer.record(source)  
            text = recognizer.recognize_google(audio_data, language=language_code)  
    return text  
  
def identify_speakers(audio_path, num_speakers=2):  
    """  
    話者識別を行う簡易版関数  
      
    Args:  
        audio_path (str): 音声ファイルのパス  
        num_speakers (int): 話者数  
      
    Returns:  
        list: 話者識別結果のリスト  
    """  
    y, sample_rate = librosa.load(audio_path, sr=None)  
    segments = []  
    window_size = len(y) // 10  
    for i in range(0, len(y), window_size):  
        end = min(i + window_size, len(y))  
        segment = y[i:end]  
        mean = np.mean(segment)  
        var = np.var(segment)  
        speaker = "話者A" if (mean + var) > 0 else "話者B"  
        segments.append({  
            "start": i / sample_rate,  
            "end": end / sample_rate,  
            "speaker": speaker  
        })  
    return segments

def add_bgm(audio_segment, bgm_path, bgm_volume):
    """
    音声にBGMを追加する関数
    
    Args:
        audio_segment (AudioSegment): 元の音声データ
        bgm_path (str): BGMファイルのパス
        bgm_volume (int): BGMの音量 (dB)
        
    Returns:
        AudioSegment: BGM追加後の音声データ
    """
    if not bgm_path or not os.path.exists(bgm_path):
        return audio_segment
        
    try:
        bgm_audio = AudioSegment.from_file(bgm_path)
        
        # BGMの長さを調整
        if len(bgm_audio) < len(audio_segment):
            # 必要な回数だけBGMを繰り返す
            repeats = int(len(audio_segment) / len(bgm_audio)) + 1
            bgm_extended = bgm_audio * repeats
            bgm_extended = bgm_extended[:len(audio_segment)]
        else:
            bgm_extended = bgm_audio[:len(audio_segment)]
        
        # BGMの音量を調整
        bgm_extended = bgm_extended.apply_gain(bgm_volume)
        
        # BGMとボイスをミックス
        result = audio_segment.overlay(bgm_extended)
        return result
    except Exception as e:
        st.error(f"BGM追加エラー: {e}")
        return audio_segment


# 編集タブの内容
with tab1:
    st.header("音声編集")
    
    if uploaded_file is not None:
        with st.spinner('音声ファイルを読み込み中...'):
            y, sample_rate, wav_path, audio_segment = load_audio(uploaded_file)
            st.session_state.waveform = y
            st.session_state.sample_rate = sample_rate
            st.session_state.audio_data = audio_segment
            
        st.success(f'音声ファイルを読み込みました: {uploaded_file.name}')
        
        # 波形の表示
        fig = plot_waveform(y, sample_rate)
        st.pyplot(fig)
        
        # 現在選択されているプリセットを取得
        preset_settings = get_preset_settings(selected_preset)
        
        if preset_settings:
            # プリセット設定を表示
            with st.expander("編集パラメータ（プリセット）"):
                st.write(f"ノイズリダクションレベル: {preset_settings['noise_reduction']}")
                st.write(f"沈黙検出閾値: {preset_settings['silence_threshold']} dB")
                st.write(f"最小沈黙時間: {preset_settings['min_silence_duration']} ms")
                st.write(f"音量ノーマライゼーション: {'有効' if preset_settings['volume_normalize'] else '無効'}")
                st.write(f"イントロ音楽: {preset_settings['intro_music']}")
                st.write(f"トランジション: {'有効' if preset_settings['add_transitions'] else '無効'}")
                st.write(f"言語: {preset_settings['language']}")
                st.write(f"BGM: {preset_settings['bgm_file'] if preset_settings['bgm_file'] else 'なし'}")
                st.write(f"BGM音量: {preset_settings['bgm_volume']} dB")

            # 編集タブ内の自動編集実行ボタンの上に追加
            with st.expander("キーワードカット設定"):
                keyword_cut_enabled = st.checkbox("キーワードカットを有効にする")
                if keyword_cut_enabled:
                    keywords_to_cut = st.text_area("カットするキーワード（1行に1つ）", 
                                                  placeholder="例: えーと\nあの\nそのー")
                    keyword_list = [k.strip() for k in keywords_to_cut.split("\n") if k.strip()]
                    keyword_padding = st.slider("キーワード前後の余白 (秒)", 0.0, 1.0, 0.1, 0.1, 
                                              help="キーワードの前後に追加でカットする時間")
                    st.info(f"カット対象: {', '.join(keyword_list) if keyword_list else 'なし'}")
            
            # 自動編集実行ボタン
            if st.button("自動編集を実行", type="primary"):
                with st.spinner('音声を編集中...'):
                    try:
                        # ノイズリダクション
                        y_reduced = reduce_noise(st.session_state.waveform, st.session_state.sample_rate, preset_settings['noise_reduction'])
                        reduced_path = os.path.join(st.session_state.temp_dir, "reduced.wav")
                        sf.write(reduced_path, y_reduced, st.session_state.sample_rate)
                        processed_audio = AudioSegment.from_file(reduced_path)
                        
                        # 音量ノーマライゼーション
                        if preset_settings['volume_normalize']:
                            processed_audio = normalize_audio(processed_audio)
                        
                        # 沈黙検出とセグメント化
                        segments = segment_audio(processed_audio, preset_settings['silence_threshold'], preset_settings['min_silence_duration'])
                        st.session_state.segments = segments
                        
                        # 効果音の追加
                        processed_audio = add_sound_effects(processed_audio, preset_settings['intro_music'], preset_settings['add_transitions'], segments)
                        
                        # BGM追加
                        if preset_settings['bgm_file'] and preset_settings['bgm_file'] in st.session_state.bgm_files:
                            bgm_path = st.session_state.bgm_files[preset_settings['bgm_file']]
                            processed_audio = add_bgm(processed_audio, bgm_path, preset_settings['bgm_volume'])
                        
                        # 処理済み音声をセッションに保存
                        st.session_state.processed_audio = processed_audio
                        
                        # 処理済み音声を一時ファイルに保存
                        processed_path = os.path.join(st.session_state.temp_dir, "processed.wav")
                        processed_audio.export(processed_path, format="wav")
                        
                        # 文字起こし（言語設定に基づく）
                        language_code = {"日本語": "ja-JP", "英語": "en-US", "スペイン語": "es-ES"}[preset_settings['language']]
                        st.session_state.transcript = transcribe_audio(processed_path, language_code)

                        # 自動編集実行ボタンの処理内（st.session_state.transcript = transcribe_audio(processed_path, language_code)の後に追加）
                        # 文字起こし後にキーワードカット処理を実行
                        if keyword_cut_enabled and keyword_list:
                            with st.spinner('キーワードをカット中...'):
                                try:
                                    # キーワードに基づいてオーディオをカット
                                    processed_audio = cut_audio_by_transcript(
                                        st.session_state.transcript, 
                                        segments, 
                                        processed_audio, 
                                        keyword_list,
                                        keyword_padding
                                    )
                                    
                                    # 処理済み音声を更新
                                    processed_path = os.path.join(st.session_state.temp_dir, "processed_keywords_cut.wav")
                                    processed_audio.export(processed_path, format="wav")
                                    st.session_state.processed_audio = processed_audio
                                    
                                    # キーワードカット後の文字起こしを再実行
                                    st.session_state.transcript = transcribe_audio(processed_path, language_code)
                                    
                                    st.success(f"{len(keyword_list)}個のキーワードをカットしました")
                                except Exception as e:
                                    st.error(f"キーワードカットエラー: {e}")
                        
                        st.success("音声の編集が完了しました！")
                    except Exception as e:
                        st.error(f"編集処理エラー: {e}")
        else:
            st.error("選択したプリセットが見つかりません。プリセットを作成してください。")
    else:
        st.info("音声ファイルをアップロードしてください。")


# プレビュータブの内容  
with tab2:  
    st.header("編集結果プレビュー")  
  
    if st.session_state.processed_audio is not None:  
        # 音声プレイヤー  
        processed_path = os.path.join(st.session_state.temp_dir, "processed.wav")  
        st.audio(processed_path)  
  
        # セグメント情報の表示  
        if st.session_state.segments:  
            with st.expander("セグメント情報"):  
                for i, (start, end) in enumerate(st.session_state.segments):  
                    start_time = datetime.timedelta(milliseconds=start)  
                    end_time = datetime.timedelta(milliseconds=end)  
                    duration = datetime.timedelta(milliseconds=end - start)  
                    st.write(f"セグメント {i+1}: {start_time} - {end_time} (長さ: {duration})")  
                      
        # 文字起こし結果の表示  
        if st.session_state.transcript:  
            with st.expander("文字起こし結果", expanded=True):  
                # 文字起こしをセグメントごとに分割して表示する  
                transcript_segments = st.session_state.transcript.split('\n')  
                for i, transcript in enumerate(transcript_segments):  
                    if i < len(st.session_state.segments):  
                        start, end = st.session_state.segments[i]  
                        start_time = datetime.timedelta(milliseconds=start)  
                        end_time = datetime.timedelta(milliseconds=end)  
                        duration = datetime.timedelta(milliseconds=end - start)  
                        st.write(f"セグメント {i+1} ({start_time} - {end_time}, 長さ: {duration}): {transcript}")  
  
        # 話者識別結果の視覚化（簡易版）  
        with st.expander("話者識別（実験的機能）"):  
            if st.button("話者識別を実行"):  
                with st.spinner('話者を識別中...'):  
                    processed_path = os.path.join(st.session_state.temp_dir, "processed.wav")  
                    speaker_segments = identify_speakers(processed_path)  
                    y, sample_rate = librosa.load(processed_path, sr=None)  
                    fig = plot_speaker_identification(y, sample_rate, speaker_segments)  
                    st.pyplot(fig)  
    else:  
        st.info("先に「編集」タブで音声を編集してください。")  


# エクスポートタブの内容
with tab3:
    st.header("編集済み音声のエクスポート")
    
    if st.session_state.processed_audio is not None:
        # 出力フォーマット選択
        output_format = st.selectbox("出力フォーマット", ["MP3", "WAV", "OGG", "MP4 (音声+静止画)"])
        
        # 出力品質設定
        if output_format == "MP3":
            quality = st.slider("MP3品質 (kbps)", 64, 320, 192, 32)
        else:
            quality = None
            
        # ファイル名設定
        default_filename = f"edited_podcast_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_filename = st.text_input("出力ファイル名", value=default_filename)
        
        # メタデータ設定
        with st.expander("メタデータ"):
            title = st.text_input("タイトル", value="ポッドキャストエピソード")
            artist = st.text_input("アーティスト/作成者", value="")
            album = st.text_input("アルバム/シリーズ", value="")
            
        # エクスポートボタン
        if st.button("エクスポート", type="primary"):
            with st.spinner('ファイルを書き出し中...'):
                try:
                    # 出力ファイルパスの設定
                    format_ext = output_format.lower() if output_format != "MP4 (音声+静止画)" else "mp4"
                    if format_ext == "mp3" or format_ext == "wav" or format_ext == "ogg":
                        output_path = os.path.join(st.session_state.temp_dir, f"{output_filename}.{format_ext}")
                        
                        # メタデータの設定
                        tags = {
                            "title": title,
                            "artist": artist,
                            "album": album
                        }
                        
                        # ファイルの出力
                        if format_ext == "mp3":
                            st.session_state.processed_audio.export(
                                output_path, 
                                format="mp3", 
                                bitrate=f"{quality}k",
                                tags=tags
                            )
                        else:
                            st.session_state.processed_audio.export(
                                output_path, 
                                format=format_ext,
                                tags=tags
                            )
                    elif format_ext == "mp4":
                        # 一時WAVファイルを作成
                        temp_wav = os.path.join(st.session_state.temp_dir, "temp_export.wav")
                        st.session_state.processed_audio.export(temp_wav, format="wav")
                        
                        # MP4に変換
                        output_path = os.path.join(st.session_state.temp_dir, f"{output_filename}.mp4")
                        create_mp4(temp_wav, output_path)
                    
                    # ダウンロードリンクの表示
                    st.success("ファイルの書き出しが完了しました！")
                    st.markdown(get_binary_file_downloader_html(output_path, 'ファイルをダウンロード'), unsafe_allow_html=True)
                    
                    # ファイルサイズの表示
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MBに変換
                    st.info(f"ファイルサイズ: {file_size:.2f} MB")
                    
                except Exception as e:
                    st.error(f"エクスポートエラー: {e}")
    else:
        st.info("先に「編集」タブで音声を編集してください。")

# アプリケーション終了時の一時ファイル削除処理
def cleanup():
    """アプリケーション終了時に一時ファイルを削除"""
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)

# Streamlitのセッション終了時にクリーンアップを実行
import atexit
atexit.register(cleanup)
