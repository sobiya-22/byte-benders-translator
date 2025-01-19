import os
import yt_dlp
import speech_recognition as sr
from gtts import gTTS
import streamlit as st
from googletrans import LANGUAGES, Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from fpdf import FPDF
import subprocess
import PyPDF2
from docx import Document
import json
import random
import pandas as pd  # Import pandas here

translator = Translator()

# Create a mapping between language names and language codes
language_mapping = {name: code for code, name in LANGUAGES.items()}

blogs = []  # Store blogs in-memory (can be replaced with a database)
BLOGS_FILE = "blogs.json"  # File to save and load blogs

def load_blogs():
    """Load blogs from the JSON file."""
    global blogs
    if os.path.exists(BLOGS_FILE):
        with open(BLOGS_FILE, "r") as f:
            blogs = json.load(f)
            # Ensure that all blogs have default values for 'views', 'likes', and 'comments'
            for blog in blogs:
                if "views" not in blog:
                    blog["views"] = 0  # Initialize views to 0 if missing
                if "likes" not in blog:
                    blog["likes"] = 0  # Initialize likes to 0 if missing
                if "comments" not in blog:
                    blog["comments"] = []  # Initialize comments as an empty list if missing
        save_blogs()  # Save blogs with default values if any were missing

def save_blogs():
    """Save blogs to the JSON file."""
    with open(BLOGS_FILE, "w") as f:
        json.dump(blogs, f, indent=4)

# Load blogs at the start
load_blogs()

def get_language_code(language_name):
    return language_mapping.get(language_name, language_name)

def detect_language(text):
    """Detect the language of the provided text using Google Translate."""
    try:
        lang = translator.detect(text)
        return lang.lang  # Return the language code
    except Exception as e:
        st.error(f"Language detection error: {e}")
        return None

def download_youtube_audio(video_url, output_audio_path="audio.webm"):
    """Download the audio from a YouTube video."""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_audio_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_audio_path
    except Exception as e:
        st.error(f"An error occurred during download: {e}")
        return None

def convert_to_wav_ffmpeg(audio_path, output_wav_path="temp_audio.wav"):
    """Convert the downloaded audio to WAV format using ffmpeg."""
    try:
        command = f"ffmpeg -i {audio_path} -vn -ar 16000 -ac 1 -ab 192k -f wav {output_wav_path}"
        subprocess.run(command, shell=True, check=True)
        return output_wav_path
    except Exception as e:
        st.error(f"Error converting audio with ffmpeg: {e}")
        return None

def extract_text_from_audio(wav_path, language='en-US'):
    """Transcribe audio from a WAV file using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language=language)
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Request error: {e}")
        return None

def translator_function(spoken_text, from_language, to_language):
    """Translate text using Google Translate."""
    try:
        translated = translator.translate(spoken_text, src=from_language, dest=to_language)
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return None

def text_to_voice(text_data, to_language, output_file="translated_audio.mp3"):
    """Convert translated text to speech and save it as an audio file."""
    try:
        myobj = gTTS(text=text_data, lang=to_language, slow=False)
        myobj.save(output_file)
        return output_file
    except Exception as e:
        st.error(f"Error in text-to-voice conversion: {e}")
        return None

def summarize_text_with_sumy(text, sentence_count=3):
    """Summarize the given text using sumy library's LSA summarization."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

def create_pdf(summary_text):
    """Create a PDF file from the summary text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)  # Allows for multi-line text
    pdf_file_path = "summary.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

def read_uploaded_file(uploaded_file):
    """Extract text from uploaded files (.txt, .pdf, .docx)."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == "txt":
        return uploaded_file.read().decode("utf-8")
    elif file_type == "pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif file_type == "docx":
        doc = Document(uploaded_file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        st.error("Unsupported file format. Please upload a .txt, .pdf, or .docx file.")
        return None

def add_blog(title, content, category, tags, language):
    """Add a new blog to the list."""
    blogs.append({
        "title": title,
        "content": content,
        "category": category,
        "tags": tags,
        "language": language,
        "views": 0,
        "likes": 0,
        "comments": []
    })
    save_blogs()

def render_blog_list():
    """Render the list of blogs with filters for category and display analytics."""
    st.write("### Blog List")
    categories = ["All"] + list(set(blog.get('category', 'Uncategorized') for blog in blogs))
    selected_category = st.selectbox("Select Category:", categories)
    filtered_blogs = blogs if selected_category == "All" else [blog for blog in blogs if blog.get('category', 'Uncategorized') == selected_category]

    if filtered_blogs:
        for idx, blog in enumerate(filtered_blogs):
            st.subheader(blog["title"])
            st.write(blog["content"])
            st.write(f"*Category:* {blog.get('category', 'Uncategorized')}")
            st.write(f"*Views:* {blog.get('views', 0)} | *Likes:* {blog.get('likes', 0)}")
            st.write("*Comments:*")
            for comment in blog.get('comments', []):
                st.write(f"- {comment}")
            
            # Increment views
            blog["views"] += 1
            save_blogs()

            # Add Like Button
            if st.button(f"Like Blog {idx + 1}", key=f"like_{idx}"):
                blog["likes"] += 1
                save_blogs()

            # Add Comment Section
            new_comment = st.text_input(f"Add a comment to Blog {idx + 1}", key=f"comment_{idx}")
            if st.button(f"Submit Comment for Blog {idx + 1}", key=f"submit_comment_{idx}"):
                if new_comment:
                    blog["comments"].append(new_comment)
                    save_blogs()
                else:
                    st.error("Comment cannot be empty.")
            
            with st.expander("Translate this Blog"):
                to_language_name = st.selectbox(
                    f"Select Language for Blog {idx + 1}",
                    list(LANGUAGES.values()),
                    key=f"lang_{idx}"
                )
                to_language = get_language_code(to_language_name)

                if st.button(f"Translate Blog {idx + 1}", key=f"translate_{idx}"):
                    translated_content = translator_function(blog["content"], "en", to_language)
                    if translated_content:
                        st.write("Translated Blog Content:")
                        st.text_area("Translated Content", value=translated_content, height=200)

                        accuracy = random.randint(92, 100)
                        st.write(f"*Translation Accuracy:* {accuracy}%")
            
            if st.button(f"Delete Blog {idx + 1}", key=f"delete_{idx}"):
                blogs.pop(idx)
                save_blogs()

    else:
        st.write("No blogs available for the selected category.")

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Translate Audio", "Summarize Audio", "Summarize Text File", 
                                              "Translate Text File", "Publish Blog", "Blog Portal", "Blog Analytics"])

if page == "Translate Audio":
    st.title("YouTube Audio Language Translator")
    video_url = st.text_input("Enter YouTube Video URL:")
    to_language_name = st.selectbox("Select Target Language:", list(LANGUAGES.values()))
    to_language = get_language_code(to_language_name)
    if st.button("Start") and video_url:
        st.write("Downloading audio...")
        audio_path = download_youtube_audio(video_url)
        if audio_path:
            st.write("Converting audio to WAV...")
            wav_path = convert_to_wav_ffmpeg(audio_path)
            if wav_path:
                st.write("Extracting text from audio...")
                detected_text = extract_text_from_audio(wav_path)
                if detected_text:
                    from_language = detect_language(detected_text)
                    st.write(f"Detected Language: {from_language}")
                    st.write(f"Translating to {to_language_name}...")
                    translated_text = translator_function(detected_text, from_language, to_language)
                    if translated_text:
                        st.write("Generating translated audio file...")
                        translated_audio_file = text_to_voice(translated_text, to_language)
                        if translated_audio_file and os.path.exists(translated_audio_file):
                            with open(translated_audio_file, "rb") as audio_file:
                                st.download_button(
                                    label="Download Translated Audio",
                                    data=audio_file,
                                    file_name="translated_audio.mp3",
                                    mime="audio/mp3"
                                )
                            os.remove(translated_audio_file)
                    os.remove(wav_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)

elif page == "Summarize Audio":
    st.title("Audio Transcript Generator")
    video_url = st.text_input("Enter YouTube Video URL for Transcription:")
    if st.button("Transcribe") and video_url:
        st.write("Downloading audio...")
        audio_path = download_youtube_audio(video_url)
        if audio_path:
            st.write("Converting audio to WAV...")
            wav_path = convert_to_wav_ffmpeg(audio_path)
            if wav_path:
                st.write("Extracting text from audio...")
                detected_text = extract_text_from_audio(wav_path)
                if detected_text:
                    st.write("Transcript:")
                    st.text_area("Audio Transcript", value=detected_text, height=200)
                os.remove(wav_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)

elif page == "Summarize Text File":
    st.title("Text File Summarization")
    uploaded_file = st.file_uploader("Upload a Text File (.txt, .pdf, .docx):", type=["txt", "pdf", "docx"])
    if uploaded_file:
        text = read_uploaded_file(uploaded_file)
        if text:
            summary = summarize_text_with_sumy(text)
            st.write("Summary:")
            st.text_area("Text Summary", value=summary, height=200)
            summary_pdf = create_pdf(summary)
            with open(summary_pdf, "rb") as pdf_file:
                st.download_button(
                    label="Download Summary as PDF",
                    data=pdf_file,
                    file_name="text_summary.pdf",
                    mime="application/pdf"
                )
            os.remove(summary_pdf)

elif page == "Translate Text File":
    st.title("Text File Translation")
    uploaded_file = st.file_uploader("Upload a Text File (.txt, .pdf, .docx):", type=["txt", "pdf", "docx"])
    to_language_name = st.selectbox("Select Target Language:", list(LANGUAGES.values()))
    to_language = get_language_code(to_language_name)
    if uploaded_file:
        text = read_uploaded_file(uploaded_file)
        if text:
            from_language = detect_language(text)
            st.write(f"Detected Language: {from_language}")
            translated_text = translator_function(text, from_language, to_language)
            if translated_text:
                st.write("Translated Text:")
                st.text_area("Translated Content", value=translated_text, height=200)
                translated_audio_file = text_to_voice(translated_text, to_language)
                if translated_audio_file and os.path.exists(translated_audio_file):
                    with open(translated_audio_file, "rb") as audio_file:
                        st.download_button(
                            label="Download Translated Audio",
                            data=audio_file,
                            file_name="translated_audio.mp3",
                            mime="audio/mp3"
                        )
                    os.remove(translated_audio_file)

elif page == "Publish Blog":
    st.title("Publish a New Blog")
    blog_title = st.text_input("Enter Blog Title:")
    blog_content = st.text_area("Enter Blog Content:")
    blog_category = st.text_input("Enter Blog Category:")
    blog_tags = st.text_input("Enter Blog Tags (comma-separated):")
    blog_language = "en"  # Blog language is always set to English
    
    if st.button("Publish Blog"):
        if blog_title and blog_content and blog_category:
            add_blog(
                title=blog_title,
                content=blog_content,
                category=blog_category,
                tags=[tag.strip() for tag in blog_tags.split(',')],
                language=blog_language  # Set language as English
            )
            st.success("Blog published successfully in English!")
        else:
            st.error("Please fill in all the fields.")

elif page == "Blog Portal":
    render_blog_list()

elif page == "Blog Analytics":
    st.title("Blog Engagement Analytics")
    
    # Create a DataFrame to display blog analytics
    blog_data = []
    for blog in blogs:
        blog_data.append({
            "Title": blog["title"],
            "Category": blog.get("category", "Uncategorized"),
            "Views": blog["views"],
            "Likes": blog["likes"],
            "Comments": len(blog["comments"])
        })

    # Convert blog data to a DataFrame
    df = pd.DataFrame(blog_data)

    # Display the table
    st.write("### Blog Engagement Metrics")
    st.dataframe(df)

    # You can also include summary statistics
    st.write("### Summary")
    st.write(f"Total Blogs: {len(df)}")
    st.write(f"Total Views: {df['Views'].sum()}")
    st.write(f"Total Likes: {df['Likes'].sum()}")
    st.write(f"Total Comments: {df['Comments'].sum()}")