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

# セッション状態の初期化関数  
def initialize_session_state():  
    if 'audio_data' not in st.session_state:  
        st.session_state.audio_data = None  
    if 'waveform' not in st.session_state:  
        st.session_state.waveform = None  
    if 'sr' not in st.session_state:  
        st.session_state.sr = None  
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
  
initialize_session_state()  
  
# アプリのタイトルとスタイル設定  
st.set_page_config(page_title="ポッドキャスト自動音声編集アプリ", layout="wide")  
st.title("ポッドキャスト自動音声編集アプリ")  
  
# 音声ファイルのアップロード関数  
def upload_audio_file():  
    st.sidebar.header("ファイルのアップロード")  
    return st.sidebar.file_uploader("音声ファイルをアップロード", type=['wav', 'mp3', 'ogg', 'flac'])  

uploaded_file = upload_audio_file()  
  
# 音声ファイル読み込み関数  
def load_audio(file):  
    try:  
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:  
            tmp_file.write(file.read())  
            tmp_path = tmp_file.name  
            try:  
                audio_segment = AudioSegment.from_file(tmp_path)  
            except Exception as e:  
                st.error(f"音声ファイルの読み込みエラー (Pydub): {e}")  
                return None, None, None, None  
  
            ext = os.path.splitext(file.name)[1][1:]  # 拡張子の取得（例: 'mp3'）  
            st.session_state.original_audio_format = ext  
  
            temp_export_path = os.path.join(st.session_state.temp_dir, "temp.wav")  
            try:  
                audio_segment.export(temp_export_path, format="wav")  
            except Exception as e:  
                st.error(f"音声ファイルのエクスポートエラー (Pydub): {e}")  
                return None, None, None, None  
  
            try:  
                y, sr = librosa.load(temp_export_path, sr=None)  
            except Exception as e:  
                st.error(f"音声ファイルの読み込みエラー (Librosa): {e}")  
                return None, None, None, None  
  
            return y, sr, temp_export_path, audio_segment  
  
    except Exception as e:  
        st.error(f"音声ファイルの読み込みエラー (全体): {e}")  
        return None, None, None, None  
  
# 波形表示関数  
def plot_waveform(y, sr):  
    fig, ax = plt.subplots(figsize=(10, 2))  
    librosa.display.waveshow(y, sr=sr, ax=ax)  
    ax.set_title('音声波形')  
    ax.set_xlabel('時間 (秒)')  
    ax.set_ylabel('振幅')  
    return fig  
  
# 音声認識と文字起こし関数  
@st.cache_data  
def transcribe_audio(audio_path, language_code):  
    try:  
        recognizer = sr.Recognizer()  
        audio_file = sr.AudioFile(audio_path)  
        with audio_file as source:  
            audio_data = recognizer.record(source)  
            text = recognizer.recognize_google(audio_data, language=language_code)  
        return text  
    except Exception as e:  
        return f"文字起こしエラー: {str(e)}"  
  
if uploaded_file is not None:  
    y, sr, wav_path, audio_segment = load_audio(uploaded_file)  
    if y is not None and sr is not None and wav_path is not None and audio_segment is not None:  
        st.session_state.audio_data = audio_segment  
        st.session_state.waveform = y  
        st.session_state.sr = sr  
  
        st.audio(wav_path)  
        fig = plot_waveform(y, sr)  
        st.pyplot(fig)  
  
# 音声処理パラメータ設定関数  
def set_audio_parameters():  
    st.sidebar.header("音声処理設定")  
    noise_reduction = st.sidebar.slider("ノイズリダクションレベル", 0.0, 1.0, 0.3, 0.1)  
    silence_threshold = st.sidebar.slider("沈黙検出閾値(dB)", -60, -20, -40, 5)  
    min_silence_duration = st.sidebar.slider("最小沈黙時間(ms)", 300, 2000, 1000, 100)  
    volume_normalize = st.sidebar.checkbox("音量ノーマライゼーション", True)  
    return noise_reduction, silence_threshold, min_silence_duration, volume_normalize  
  
noise_reduction, silence_threshold, min_silence_duration, volume_normalize = set_audio_parameters()  
  
# 効果音設定関数  
def set_sound_effects():  
    st.sidebar.header("効果音設定")  
    intro_music = st.sidebar.selectbox("イントロ音楽", ["なし", "プロフェッショナル", "アップビート", "リラックス"])  
    add_transitions = st.sidebar.checkbox("セグメント間にトランジションを追加", False)  
    return intro_music, add_transitions  
  
intro_music, add_transitions = set_sound_effects()  
  
# 言語設定関数  
def set_language():  
    st.sidebar.header("言語設定")  
    language = st.sidebar.selectbox("音声認識言語", ["日本語", "英語", "スペイン語"])  
    return language  
  
language = set_language()  
  
language_code = {  
    "日本語": "ja-JP",  
    "英語": "en-US",  
    "スペイン語": "es-ES"  
}  
    
# セグメント化関数  
def segment_audio(audio_segment, silence_thresh, min_silence_len):  
    silence_parts = silence.detect_silence(  
        audio_segment,   
        min_silence_len=min_silence_len,   
        silence_thresh=silence_thresh  
    )  
    segments = []  
    prev_end = 0  
    for start, end in silence_parts:  
        if start > prev_end:  
            segments.append((prev_end, start))  
        prev_end = end  
    if prev_end < len(audio_segment):  
        segments.append((prev_end, len(audio_segment)))  
    return segments  
  
# ノイズリダクション関数  
def reduce_noise(y, sr, reduction_amount):  
    y_reduced = y.copy()  
    noise_mask = np.abs(y) < (reduction_amount * np.max(np.abs(y)))  
    y_reduced[noise_mask] = 0  
    return y_reduced  
  
# 音量ノーマライゼーション関数  
def normalize_audio(audio_segment):  
    return effects.normalize(audio_segment)  
  
# 効果音追加関数  
def add_sound_effects(audio_segment, intro_type, add_transitions, segments):  
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
  
# MP4出力関数  
def create_mp4(audio_path, output_path):  
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
  
# ダウンロードリンク作成関数  
def get_binary_file_downloader_html(bin_file, file_label='File'):  
    with open(bin_file, 'rb') as f:  
        data = f.read()  
    bin_str = base64.b64encode(data).decode()  
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'  
    return href  
  
# タブの作成  
tab1, tab2, tab3, tab4, tab5 = st.tabs(["音声分析", "編集", "プレビュー", "エクスポート", "プロジェクト管理"])  
  
# ファイルが選択されたら処理開始  
def process_uploaded_file(uploaded_file):  
    if uploaded_file is not None:  
        try:  
            with st.spinner('音声ファイルを読み込んでいます...'):  
                y, sr, wav_path, audio_segment = load_audio(uploaded_file)  
                st.session_state.audio_data = audio_segment  
                st.session_state.waveform = y  
                st.session_state.sr = sr  
                return wav_path  
        except Exception as e:  
            st.error(f"音声ファイルの処理中にエラーが発生しました: {e}")  
  
wav_path = process_uploaded_file(uploaded_file)  
  
# 各タブの処理をリファクタリング  
with tab1:  
    if uploaded_file is not None:  
        st.subheader("音声分析")  
        st.audio(wav_path)  
        st.write("波形分析")  
        fig = plot_waveform(st.session_state.waveform, st.session_state.sr)  
        st.pyplot(fig)  
        if st.button("文字起こしを実行"):  
            with st.spinner('文字起こしを実行中...'):  
                text = transcribe_audio(wav_path, language_code[language])  
                st.session_state.transcript = text  
                st.write("文字起こし結果:")  
                st.text_area("テキスト", value=text, height=200)  
        if st.button("話者識別を実行"):  
            with st.spinner("話者識別を実行中..."):  
                speaker_segments = identify_speakers(wav_path)  
                st.write("話者識別結果:")  
                for i, segment in enumerate(speaker_segments):  
                    st.write(f"{segment['speaker']}: {segment['start']:.2f}秒 - {segment['end']:.2f}秒")  
                fig, ax = plt.subplots(figsize=(10, 3))  
                colors = {"話者A": "blue", "話者B": "red"}  
                for segment in speaker_segments:  
                    start = segment["start"]  
                    end = segment["end"]  
                    speaker = segment["speaker"]  
                    ax.axvspan(start, end, alpha=0.3, color=colors[speaker])  
                librosa.display.waveshow(st.session_state.waveform, sr=st.session_state.sr, ax=ax)  
                ax.set_title("話者識別結果")  
                ax.legend(colors.keys())  
                st.pyplot(fig)  
                  
with tab2:  
    st.subheader("音声編集")  
      
    if st.button("音声を自動処理"):  
        try:  
            with st.spinner('処理中...'):  
                y_reduced = reduce_noise(st.session_state.waveform, st.session_state.sr, noise_reduction)  
                noise_reduced_path = os.path.join(st.session_state.temp_dir, f"noise_reduced.wav")  
                sf.write(noise_reduced_path, y_reduced, st.session_state.sr)  
                processed_audio = AudioSegment.from_file(noise_reduced_path)  
                st.session_state.audio_data = processed_audio  # ここで session_state の audio_data を更新する  
                if volume_normalize:  
                    processed_audio = normalize_audio(processed_audio)  
                segments = segment_audio(processed_audio, silence_threshold, min_silence_duration)  
                st.session_state.segments = segments  
                final_audio = add_sound_effects(processed_audio, intro_music, add_transitions, segments)  
                processed_path = os.path.join(st.session_state.temp_dir, f"processed.{st.session_state.original_audio_format}")  
                final_audio.export(processed_path, format=st.session_state.original_audio_format)  
                st.session_state.processed_audio = processed_path  
                st.success("処理が完了しました！")  
        except Exception as e:  
            st.error(f"音声処理中にエラーが発生しました: {str(e)}")  
      
    # キーワードを入力してカット  
    keywords_to_cut = st.text_input("カットするキーワード（カンマ区切りで複数指定可能）").split(',')  
    if st.button("キーワードでカット"):  
        try:  
            with st.spinner('キーワードでカット中...'):  
                final_audio = cut_audio_by_transcript(st.session_state.transcript, st.session_state.segments, st.session_state.audio_data, keywords_to_cut, st.session_state.sr)  
                processed_path = os.path.join(st.session_state.temp_dir, f"cut_processed.wav")  
                final_audio.export(processed_path, format=st.session_state.original_audio_format)  
                st.session_state.processed_audio = processed_path  
                st.success("キーワードでカットが完了しました！")  
        except Exception as e:  
            st.error(f"キーワードカット中にエラーが発生しました: {str(e)}")  
  
with tab3:  
    st.subheader("プレビュー")  
    if st.session_state.processed_audio:  
        st.write("処理後の音声:")  
        st.audio(st.session_state.processed_audio)  
        st.write("処理前後の波形比較")  
        y_processed, sr_processed = librosa.load(st.session_state.processed_audio, sr=None)  
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))  
        librosa.display.waveshow(st.session_state.waveform, sr=st.session_state.sr, ax=ax[0])  
        ax[0].set_title('処理前')  
        librosa.display.waveshow(y_processed, sr=sr_processed, ax=ax[1])  
        ax[1].set_title('処理後')  
        st.pyplot(fig)  
        st.write("スペクトログラム分析")  
        fig, ax = plt.subplots(figsize=(10, 4))  
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_processed)), ref=np.max)  
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr_processed, ax=ax)  
        fig.colorbar(img, ax=ax, format="%+2.0f dB")  
        st.pyplot(fig)  
  
with tab4:  
    st.subheader("エクスポート")  
    if st.session_state.processed_audio:  
        col1, col2 = st.columns(2)  
        with col1:  
            export_quality = st.select_slider(  
                "出力音質",  
                options=["低 (96k)", "中 (192k)", "高 (320k)"],  
                value="中 (192k)"  
            )  
            quality_map = {  
                "低 (96k)": "96k",  
                "中 (192k)": "192k",  
                "高 (320k)": "320k"  
            }  
            selected_quality = quality_map[export_quality]  
        with col2:  
            export_format = st.radio(  
                "出力形式",  
                ["MP4", "MP3", "WAV"],  
                horizontal=True  
            )  
        thumbnail_upload = st.file_uploader("サムネイル画像 (オプション)", type=['jpg', 'jpeg', 'png'])  
        thumbnail_path = None  
        if thumbnail_upload:  
            thumbnail_path = os.path.join(st.session_state.temp_dir, "thumbnail.png")  
            with open(thumbnail_path, "wb") as f:  
                f.write(thumbnail_upload.getbuffer())  
            st.image(thumbnail_path, width=300)  
        metadata = st.text_area("メタデータ (タイトル、説明など)", "タイトル: マイポッドキャスト\n作成者: ポッドキャスト編集者\n説明: このポッドキャストは自動編集アプリで作成されました。")  
        if st.button(f"{export_format}にエクスポート"):  
            with st.spinner(f'{export_format}に変換中...'):  
                if export_format == "MP4":  
                    output_file = os.path.join(st.session_state.temp_dir, f"podcast_output.{export_format.lower()}")  
                    image_src = thumbnail_path if thumbnail_path else os.path.join(st.session_state.temp_dir, "placeholder.png")  
                    if not os.path.exists(image_src):  
                        plt.figure(figsize=(16, 9))  
                        plt.text(0.5, 0.5, 'ポッドキャスト', horizontalalignment='center', verticalalignment='center', fontsize=40)  
                        plt.axis('off')  
                        plt.savefig(image_src, bbox_inches='tight', pad_inches=0)  
                        plt.close()  
                    cmd = [  
                        'ffmpeg', '-y',  
                        '-loop', '1',  
                        '-i', image_src,  
                        '-i', st.session_state.processed_audio,  
                        '-c:v', 'libx264',  
                        '-tune', 'stillimage',  
                        '-c:a', 'aac',  
                        '-b:a', selected_quality,  
                        '-pix_fmt', 'yuv420p',  
                        '-shortest',  
                        output_file  
                    ]  
                    subprocess.run(cmd, check=True)  
                else:  
                    output_file = os.path.join(st.session_state.temp_dir, f"podcast_output.{export_format.lower()}")  
                    audio = AudioSegment.from_file(st.session_state.processed_audio)  
                    export_params = {}  
                    if export_format == "MP3":  
                        export_params["bitrate"] = selected_quality  
                    audio.export(output_file, format=export_format.lower(), **export_params)  
                st.markdown(get_binary_file_downloader_html(output_file, f'ポッドキャスト.{export_format.lower()}'), unsafe_allow_html=True)  
                st.success(f"{export_format}エクスポートが完了しました！")  
  
with tab5:  
    st.subheader("プロジェクト管理")  
    col1, col2 = st.columns(2)  
    with col1:  
        st.write("プロジェクトの保存")  
        project_name = st.text_input("プロジェクト名", "マイポッドキャスト")  
        if st.button("プロジェクトを保存"):  
            if st.session_state.audio_data:  
                save_path = save_project(project_name)  
                st.success(f"プロジェクトを保存しました: {save_path}")  
            else:  
                st.warning("音声データがありません。まずは音声ファイルをアップロードしてください。")  
    with col2:  
        st.write("プロジェクトの読み込み")  
        project_dir = os.path.join(os.path.expanduser("~"), "podcast_editor_projects")  
        os.makedirs(project_dir, exist_ok=True)  
        project_files = [f for f in os.listdir(project_dir) if f.endswith(".json")]  
        if project_files:  
            selected_project = st.selectbox("プロジェクトを選択", project_files)  
            if st.button("プロジェクトを読み込む"):  
                project_path = os.path.join(project_dir, selected_project)  
                load_project(project_path)  
                st.success(f"プロジェクト '{selected_project}' を読み込みました")  
                st.experimental_rerun()  
        else:  
            st.info("保存されたプロジェクトがありません。")  
  
def identify_speakers(audio_path, num_speakers=2):  
    y, sr = librosa.load(audio_path, sr=None)  
    segments = []  
    window_size = len(y) // 10  
    for i in range(0, len(y), window_size):  
        end = min(i + window_size, len(y))  
        segment = y[i:end]  
        mean = np.mean(segment)  
        var = np.var(segment)  
        speaker = "話者A" if (mean + var) > 0 else "話者B"  
        segments.append({  
            "start": i / sr,  
            "end": end / sr,  
            "speaker": speaker  
        })  
    return segments  
  
def save_project(project_name):  
    project_dir = os.path.join(os.path.expanduser("~"), "podcast_editor_projects")  
    os.makedirs(project_dir, exist_ok=True)  
    project_file = os.path.join(project_dir, f"{project_name}.json")  
    project_audio = os.path.join(project_dir, f"{project_name}_audio.wav")  
    if st.session_state.audio_data:  
        st.session_state.audio_data.export(project_audio, format="wav")  
    project_info = {  
        "name": project_name,  
        "created_at": datetime.datetime.now().isoformat(),  
        "audio_path": project_audio,  
        "settings": {  
            "noise_reduction": noise_reduction,  
            "silence_threshold": silence_threshold,  
            "min_silence_duration": min_silence_duration,  
            "volume_normalize": volume_normalize,  
            "intro_music": intro_music,  
            "add_transitions": add_transitions,  
            "language": language  
        },  
        "segments": [{"start": s[0], "end": s[1]} for s in st.session_state.segments],  
        "transcript": st.session_state.transcript  
    }  
    with open(project_file, "w", encoding="utf-8") as f:  
        json.dump(project_info, f, ensure_ascii=False, indent=2)  
    return project_file  
  
def load_project(project_file):  
    with open(project_file, "r", encoding="utf-8") as f:  
        project_info = json.load(f)  
    audio_path = project_info["audio_path"]  
    if os.path.exists(audio_path):  
        st.session_state.audio_data = AudioSegment.from_file(audio_path)  
        y, sr = librosa.load(audio_path, sr=None)  
        st.session_state.waveform = y  
        st.session_state.sr = sr  
    settings = project_info["settings"]  
    st.sidebar.slider("ノイズリダクションレベル", 0.0, 1.0, settings["noise_reduction"], 0.1)  
    st.sidebar.slider("沈黙検出閾値(dB)", -60, -20, settings["silence_threshold"], 5)  
    st.sidebar.slider("最小沈黙時間(ms)", 300, 2000, settings["min_silence_duration"], 100)  
    st.sidebar.checkbox("音量ノーマライゼーション", settings["volume_normalize"])  
    st.sidebar.selectbox("イントロ音楽", ["なし", "プロフェッショナル", "アップビート", "リラックス"], index=["なし", "プロフェッショナル", "アップビート", "リラックス"].index(settings["intro_music"]))  
    st.sidebar.checkbox("セグメント間にトランジションを追加", settings["add_transitions"])  
    st.sidebar.selectbox("音声認識言語", ["日本語", "英語", "スペイン語"], index=["日本語", "英語", "スペイン語"].index(settings["language"]))  
    st.session_state.segments = [(s["start"], s["end"]) for s in project_info["segments"]]  
    st.session_state.transcript = project_info["transcript"]  
    return project_info  
  
def identify_speakers(audio_path, num_speakers=2):  
    y, sr = librosa.load(audio_path, sr=None)  
    segments = []  
    window_size = len(y) // 10  
    for i in range(0, len(y), window_size):  
        end = min(i + window_size, len(y))  
        segment = y[i:end]  
        mean = np.mean(segment)  
        var = np.var(segment)  
        speaker = "話者A" if (mean + var) > 0 else "話者B"  
        segments.append({  
            "start": i / sr,  
            "end": end / sr,  
            "speaker": speaker  
        })  
    return segments  
  
def cleanup_temp_files():  
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):  
        try:  
            shutil.rmtree(st.session_state.temp_dir)  
        except Exception as e:  
            pass  

# 音声ファイルから指定された文字部分の音声部分をカットする関数  
def cut_audio_by_transcript(transcript, segments, audio_segment, keywords, sr):  
    for keyword in keywords:  
        for start, end in get_keyword_timestamps(transcript, segments, keyword, sr):  
            audio_segment = audio_segment[:start * 1000] + audio_segment[end * 1000:]  
    return audio_segment  
  
# キーワードが見つかる時間（秒）範囲を返す関数  
def get_keyword_timestamps(transcript, segments, keyword, sr):  
    timestamps = []  
    start_time = 0  
    for segment in segments:  
        start, end = segment  
        text = transcribe_audio_partial(st.session_state.audio_data, language_code, start, end, sr)  
        if keyword in text:  
            keyword_start = start_time + (text.find(keyword) / len(text)) * (end - start) / sr  
            keyword_end = keyword_start + (len(keyword) / len(text)) * (end - start) / sr  
            timestamps.append((keyword_start, keyword_end))  
        start_time += (end - start) / sr  
    return timestamps  
  
# サブセグメントの文字起こしを行う部分関数  
def transcribe_audio_partial(audio_segment, language_code, start, end, sr):  
    partial_audio = audio_segment[start:end]  
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:  # 一時的にWAV形式で保存  
        partial_audio.export(tmp_file.name, format="wav")  
        recognizer = sr.Recognizer()  
        audio_file = sr.AudioFile(tmp_file.name)  
        with audio_file as source:  
            audio_data = recognizer.record(source)  
            text = recognizer.recognize_google(audio_data, language=language_code)  
    return text  
  
# アプリのフッター情報  
st.markdown("---")  
st.write("© 2025 ポッドキャスト自動音声編集アプリ")  
st.write("Python + Streamlit + FFmpeg で作成")  
  
# アプリの終了時にクリーンアップ  
import atexit  
atexit.register(cleanup_temp_files)  
  
# ユーザーマニュアルを追加  
with st.expander("使い方ガイド"):  
    st.write("""  
    ### ポッドキャスト自動音声編集アプリの使い方  
      
    1. **音声ファイルのアップロード**  
       - サイドバーの「ファイルのアップロード」セクションから音声ファイルをアップロードします。  
       - WAV, MP3, OGG, FLACフォーマットに対応しています。  
      
    2. **音声分析**  
       - 「音声分析」タブで波形を確認します。  
       - 「文字起こしを実行」ボタンをクリックして自動文字起こしを行います。  
       - 「話者識別を実行」ボタンで話者の区別を試みます。  
      
    3. **音声編集**  
       - 「編集」タブの「音声を自動処理」ボタンで自動編集を実行します。  
       - 編集設定はサイドバーで調整可能です。  
       - セグメント情報が表示され、個別にカスタマイズできます。  
       - 「高度なオプション」でさらに詳細な編集が可能です。  
      
    4. **プレビュー**  
       - 「プレビュー」タブで処理結果を確認します。  
       - 処理前後の波形比較やスペクトログラム分析を行えます。  
      
    5. **エクスポート**  
       - 「エクスポート」タブでMP4/MP3/WAVとして出力できます。  
       - 音質やサムネイル画像を設定できます。  
      
    6. **プロジェクト管理**  
       - 「プロジェクト管理」タブでプロジェクトの保存と読み込みができます。  
       - 設定や編集状態を保存して後で続きを編集できます。  
    """)  
  
# エラーハンドリング  
try:  
    pass  # アプリのメイン処理はすでに実行済み  
except Exception as e:  
    st.error(f"エラーが発生しました: {str(e)}")  
    st.error("アプリを再読み込みしてください。")  
