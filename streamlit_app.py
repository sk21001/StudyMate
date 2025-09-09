import streamlit as st
import os
import io
import tempfile
import base64
import re
import requests
from collections import Counter
from dotenv import load_dotenv
import PyPDF2
from gtts import gTTS
from googletrans import Translator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
try:
    from streamlit_mic_recorder import mic_recorder
    import speech_recognition as sr
    HAS_VOICE = True
except Exception:
    HAS_VOICE = False
import subprocess
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

st.set_page_config(page_title="PDF Analyzer — Unified", layout="wide")
load_dotenv()
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
URL = os.getenv("URL", "https://eu-de.ml.cloud.ibm.com")
MODEL_ID = os.getenv("MODEL_ID", "ibm/granite-3-3-8b-instruct")

LANGUAGES = {"English":"en","Hindi":"hi","French":"fr","German":"de","Spanish":"es"}
translator = Translator()

if "doc_text" not in st.session_state: st.session_state.doc_text = ""
if "summary" not in st.session_state: st.session_state.summary = ""
if "entities" not in st.session_state: st.session_state.entities = ""
if "qa_history" not in st.session_state: st.session_state.qa_history = []
if "tts_input" not in st.session_state: st.session_state.tts_input = ""

def extract_text_from_pdf(file) -> str:
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for p in reader.pages:
            text += (p.extract_text() or "") + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""

def get_iam_token(api_key: str) -> str:
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    r = requests.post(url, headers=headers, data=data)
    if r.status_code != 200:
        st.error(f"IAM Auth Error {r.status_code}: {r.text}")
        raise RuntimeError(r.text)
    return r.json()["access_token"]

def watsonx_generate(prompt: str, max_new_tokens: int = 500) -> str:
    token = get_iam_token(API_KEY)
    endpoint = f"{URL}/ml/v1/text/generation?version=2023-05-29"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "input": prompt,
        "parameters": {"decoding_method": "greedy", "max_new_tokens": max_new_tokens, "min_new_tokens": 30},
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID
    }
    r = requests.post(endpoint, headers=headers, json=payload)
    if r.status_code != 200:
        st.error(f"Watsonx Error {r.status_code}: {r.text}")
        raise RuntimeError(r.text)
    return r.json()["results"][0]["generated_text"]

def simple_keywords(text: str, topn: int = 25):
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    stop_words = set('''a the and of to in for on with this that from by an or as is are was were be been being it its into your our their you we he she they them his her at not but if then than about over under which who whom where when how why can may might should could would will shall do does did done have has had having more most less least one two three four five six seven eight nine ten per'''.split())
    words = [w for w in words if w not in stop_words and len(w)>2]
    return Counter(words).most_common(topn)

def bar_keywords(counts):
    labels = [w for w,_ in counts]
    vals = [c for _,c in counts]
    fig = plt.figure(figsize=(12,5))
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.title("Keyword Frequency")
    st.pyplot(fig)

def draw_wordcloud(text: str):
    wc = WordCloud(width=900, height=400, background_color="white").generate(text)
    fig = plt.figure(figsize=(12,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)

def export_report_pdf(summary: str, qa_history: list) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    elems = [Paragraph("PDF Analyzer Report", styles["Title"]), Spacer(1, 0.2*inch),
             Paragraph("Summary", styles["Heading2"])]
    if summary:
        for para in summary.split("\n"):
            if para.strip():
                elems.append(Paragraph(para.replace("•","-"), styles["BodyText"]))
                elems.append(Spacer(1, 4))
    else:
        elems.append(Paragraph("No summary generated yet.", styles["BodyText"]))
    elems.append(Spacer(1, 0.2*inch))
    elems.append(Paragraph("Q&A History", styles["Heading2"]))
    if qa_history:
        for i, item in enumerate(qa_history, 1):
            q = item.get("q","").strip()
            a = item.get("a","").strip()
            elems.append(Paragraph(f"Q{i}: {q}", styles["BodyText"]))
            elems.append(Spacer(1, 2))
            for para in a.split("\n"):
                if para.strip():
                    elems.append(Paragraph(para, styles["BodyText"]))
                    elems.append(Spacer(1, 2))
            elems.append(Spacer(1,6))
    else:
        elems.append(Paragraph("No questions asked yet.", styles["BodyText"]))
    doc.build(elems)
    buf.seek(0)
    return buf.read()

def convert_to_wav(audio_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_in:
        temp_in.write(audio_bytes)
        temp_in.flush()
        temp_out = temp_in.name.replace(".webm", ".wav")
        cmd = ["ffmpeg", "-y", "-i", temp_in.name, "-ar", "16000", "-ac", "1", temp_out]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(temp_out, "rb") as f: wav_bytes = f.read()
    return wav_bytes

def ensure_env():
    missing = [k for k,v in {"API_KEY":API_KEY,"PROJECT_ID":PROJECT_ID,"URL":URL,"MODEL_ID":MODEL_ID}.items() if not v]
    if missing:
        st.sidebar.error("Missing env vars: " + ", ".join(missing))
        st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    ensure_env()
    st.caption(f"Model: {MODEL_ID}")

st.title("📑 PDF Analyzer — Unified")
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    file_bytes = uploaded.read()
    text = extract_text_from_pdf(io.BytesIO(file_bytes))
    if text: st.session_state.doc_text = text
    else: st.warning("No text extracted from PDF.")

if st.session_state.doc_text:
    st.subheader("📖 Extracted Text (search enabled)")
    search_query = st.text_input("🔍 Search in text")
    display_text = st.session_state.doc_text
    if search_query:
        display_text = display_text.replace(search_query, f"<mark style='background-color:lightblue'>{search_query}</mark>")
    st.markdown(f"<div style='white-space: pre-wrap'>{display_text}</div>", unsafe_allow_html=True)

    st.markdown("### 🔊 Read Aloud & Translate (Text-to-Speech)")
    tts_text = st.text_area("Select text to read aloud or translate", value=st.session_state.tts_input, key="tts_input")
    col1, col2 = st.columns(2)
    with col1:
        src_lang_name = st.selectbox("Original Language", list(LANGUAGES.keys()), index=0, key="src_lang")
    with col2:
        tgt_lang_name = st.selectbox("Language for Speech", list(LANGUAGES.keys()), index=0, key="tgt_lang")

    src_lang_code = LANGUAGES[src_lang_name]
    tgt_lang_code = LANGUAGES[tgt_lang_name]

    if st.button("🌐 Translate"):
        if tts_text.strip():
            try:
                translated = translator.translate(tts_text.strip(), src=src_lang_code, dest=tgt_lang_code)
                st.write(f"**Translation ({tgt_lang_name}):**")
                st.success(translated.text)
            except Exception as e:
                st.error(f"Translation failed: {e}")
        else:
            st.warning("Please enter text to translate.")

    if st.button("▶️ Read aloud"):
        try:
            display_tts_text = tts_text.strip()
            if (src_lang_code != tgt_lang_code) and tts_text.strip():
                translated = translator.translate(tts_text.strip(), src=src_lang_code, dest=tgt_lang_code)
                display_tts_text = translated.text
            if display_tts_text:
                tts = gTTS(display_tts_text, lang=tgt_lang_code)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    fp.seek(0)
                    audio_bytes = fp.read()
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button("⬇️ Download Speech MP3", data=audio_bytes, file_name="tts_output.mp3", mime="audio/mp3")
            else:
                st.warning("Please enter text to convert to speech.")
        except Exception as e:
            st.error(f"Text-to-speech failed: {e}")

    tabs = st.tabs(["Summary", "Q&A", "Keywords", "Entities", "Export"])

    with tabs[0]:
        summarize_scope = st.selectbox("Summarize scope", ["Whole document", "First 5 pages", "First 10 pages"])
        scope_text = st.session_state.doc_text
        if summarize_scope != "Whole document":
            pages = scope_text.split("\f")
            scope_text = "\n".join(pages[:5 if summarize_scope=="First 5 pages" else 10])
        if st.button("✨ Summarize with Watsonx"):
            with st.spinner("Summarizing..."):
                prompt = f"""You are an expert document analyst.
Summarize the content into 6-12 concise bullet points.
If the document includes figures, dates, or names, include them.
Keep the tone neutral and informative.
Document:
{scope_text}
"""
                try:
                    out = watsonx_generate(prompt, max_new_tokens=600)
                    st.session_state.summary = out
                    st.success("Summary ready.")
                    st.write(out)
                except Exception as e:
                    st.error(f"Summary failed: {e}")
        if st.session_state.summary:
            st.write(st.session_state.summary)

    with tabs[1]:
        q = st.text_input("Ask a question about the PDF")
        if HAS_VOICE:
            st.caption("Or record your question:")
            audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="⏹ Stop", key="mic")
            if audio and isinstance(audio, dict):
                try:
                    wav_bytes = convert_to_wav(audio["bytes"])
                    r = sr.Recognizer()
                    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
                        audio_data = r.record(source)
                    try:
                        q_from_voice = r.recognize_google(audio_data)
                        st.success(f"Recognized: {q_from_voice}")
                        if not q:
                            q = q_from_voice
                    except Exception as e:
                        st.warning(f"Voice recognition failed: {e}")
                except Exception as e:
                    st.warning(f"Audio processing error: {e}")
        if st.button("💡 Get Answer"):
            if not q.strip():
                st.warning("Please type or record a question.")
            else:
                with st.spinner("Thinking..."):
                    prompt = f"""Answer the user's question strictly using the following document.
If the answer is not present in the document, state that you cannot find it.
Provide a concise, direct answer.
Document:
{st.session_state.doc_text}
Question: {q}
"""
                    try:
                        a = watsonx_generate(prompt, max_new_tokens=500)
                        st.write(a)
                        st.session_state.qa_history.append({"q": q, "a": a})
                    except Exception as e:
                        st.error(f"Q&A failed: {e}")
        if st.session_state.qa_history:
            st.markdown("---")
            st.subheader("History")
            for i, item in enumerate(st.session_state.qa_history,1):
                st.markdown(f"**Q{i}:** {item['q']}")
                st.markdown(f"**A{i}:** {item['a']}")

    with tabs[2]:
        counts = simple_keywords(st.session_state.doc_text, topn=25)
        colk1,colk2 = st.columns(2)
        with colk1:
            st.subheader("Top keywords")
            for w,c in counts:
                st.write(f"- {w}: {c}")
        with colk2:
            st.subheader("Bar chart")
            bar_keywords(counts)
        st.subheader("Word cloud")
        draw_wordcloud(st.session_state.doc_text)

    with tabs[3]:
        if st.button("🔎 Extract Entities (Watsonx)"):
            with st.spinner("Analyzing entities..."):
                prompt = f"""Extract named entities from the document and group them as:
- People
- Organizations
- Locations
- Dates
- Laws/Policies
- Other
Return each group as a short bulleted list.
Document:
{st.session_state.doc_text}
"""
                try:
                    ents = watsonx_generate(prompt, max_new_tokens=500)
                    st.session_state.entities = ents
                    st.write(ents)
                except Exception as e:
                    st.error(f"Entity extraction failed: {e}")
        if st.session_state.entities:
            st.write(st.session_state.entities)

    with tabs[4]:
        st.write("Download your results as a PDF containing the summary and the full Q&A history.")
        if st.button("📥 Generate PDF"):
            try:
                pdf_bytes = export_report_pdf(st.session_state.summary, st.session_state.qa_history)
                st.download_button("⬇️ Download Report PDF", data=pdf_bytes, file_name="pdf_analyzer_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Export failed: {e}")
else:
    st.info("Upload a PDF to begin.")
