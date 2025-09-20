# This script requires Streamlit, NLTK, NumPy, and scikit-learn.
# Install them with: pip install -r requirements.txt
import streamlit as st
import os
import datetime
import zipfile
import io
import heapq
import logging

# Attempt to import scientific computing libraries and provide a friendly error if they are missing.
# This check ensures the app doesn't crash and guides the user to install dependencies.
try:
    from fpdf import FPDF
    from docx import Document
    import fitz  # PyMuPDF
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    import numpy as np
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError as e:
    st.title("Text Document Summarizer")
    st.error(f"A required library is missing: '{e.name}'")
    st.warning(
        "To continue, please install the required packages by running the "
        "following command in your terminal:"
    )
    st.code("pip install -r requirements.txt", language="bash")
    st.stop()  # Pause the app until dependencies are installed

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")

# --- Global Logger ---
_LOGGER = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles reading and processing of uploaded documents."""
    def read_text_file(self, uploaded_file):
        try:
            raw_bytes = uploaded_file.read()
            try:
                return raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return raw_bytes.decode('latin-1')
        except Exception as e:
            st.error(f"Error reading TXT file: {e}")
            return ""

    def read_docx_file(self, uploaded_file):
        try:
            document = Document(uploaded_file)
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
            return ""

    def read_pdf_file(self, uploaded_file):
        try:
            file_bytes = uploaded_file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                return "".join(page.get_text() for page in doc)
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""

    def process_uploaded_files(self, uploaded_files):
        docs_to_process = []
        MAX_FILE_SIZE_IN_ZIP = 10 * 1024 * 1024
        SUPPORTED_EXTENSIONS = ('.docx', '.txt', '.pdf')

        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            if file_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "text/plain", "application/pdf"
            ]:
                docs_to_process.append(uploaded_file)
            elif file_type == "application/zip":
                try:
                    with zipfile.ZipFile(uploaded_file) as z:
                        for member in z.infolist():
                            if member.filename.endswith(SUPPORTED_EXTENSIONS) and not member.is_dir():
                                if member.file_size > MAX_FILE_SIZE_IN_ZIP:
                                    st.warning(f"Skipping '{member.filename}' in '{uploaded_file.name}' because it is too large.")
                                    continue
                                with z.open(member) as inner_file:
                                    file_in_memory = io.BytesIO(inner_file.read())
                                    file_in_memory.name = os.path.basename(member.filename)
                                    docs_to_process.append(file_in_memory)
                except zipfile.BadZipFile:
                    st.error(f"The uploaded ZIP file '{uploaded_file.name}' is corrupted.")
                except Exception as e:
                    st.error(f"An error occurred while processing '{uploaded_file.name}': {e}")
            else:
                st.warning(f"Unsupported file type ('{uploaded_file.name}'). This file will be ignored.")
        return docs_to_process

class Summarizer:
    """Handles the text summarization logic."""
    def summarize_text(self, text, num_sentences=3):
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        stop_words = set(stopwords.words("english"))
        vectorizer = TfidfVectorizer(stop_words=list(stop_words))

        try:
            preprocessed_sentences = [s.lower() for s in sentences]
            sentence_vectors = vectorizer.fit_transform(preprocessed_sentences)
        except ValueError:
            return "Could not generate a summary. The document may not contain enough meaningful content."

        similarity_matrix = cosine_similarity(sentence_vectors, sentence_vectors)
        scores = np.zeros(len(sentences))
        damping_factor = 0.85
        epsilon = 1.0e-5
        iterations = 100

        for _ in range(iterations):
            prev_scores = np.copy(scores)
            for i in range(len(sentences)):
                summation = sum(
                    (similarity_matrix[i][j] * prev_scores[j]) / (sum(similarity_matrix[j]) or 1)
                    for j in range(len(sentences)) if i != j
                )
                scores[i] = (1 - damping_factor) + damping_factor * summation
            if np.allclose(prev_scores, scores, atol=epsilon):
                break

        summary_indices = heapq.nlargest(num_sentences, range(len(scores)), key=scores.__getitem__)
        summary_indices.sort()
        summary = " ".join([sentences[i] for i in summary_indices])
        return summary if summary else "Could not generate a summary."

class Exporter:
    """Handles exporting data to different formats."""
    def create_pdf_from_text(self, text):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        encoded_text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, encoded_text)
        # The fpdf.output() method returns a bytearray, but st.download_button
        # expects bytes. We convert it here to prevent the TypeError.
        return bytes(pdf.output())

    def get_download_data(self, summary_text, output_format):
        if output_format == "PDF":
            return (
                self.create_pdf_from_text(summary_text),
                "pdf",
                "application/pdf"
            )
        else:  # Text (.txt)
            return (
                summary_text.encode('utf-8'),
                "txt",
                "text/plain"
            )

    def generate_report(self, original_texts, summaries, filenames):
        report_filename = os.path.join("reports", f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_filename, "w") as f:
            f.write("--- Multi-Document Summary Report ---\n\n")
            f.write(f"Number of docs checked: {st.session_state.usage_count}\n")
            f.write(f"Billing summary: {st.session_state.usage_count * 10} INR\n")
            f.write("\n" + "="*40 + "\n")

            for i, (text, summary, filename) in enumerate(zip(original_texts, summaries, filenames)):
                f.write(f"\n--- Document {i+1}: {filename} ---\n")
                f.write("\n--- Original Text ---\n")
                f.write(text)
                f.write("\n\n--- Generated Summary ---\n")
                f.write(summary)
                f.write("\n" + "="*40 + "\n")
        return report_filename

    def generate_pdf_report(self, original_texts, summaries, filenames):
        """Generates a PDF report and returns its path and content as bytes."""
        report_filename = os.path.join("reports", f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        doc = SimpleDocTemplate(report_filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Smart Doc Checker Report", styles['h1']))
        story.append(Spacer(1, 0.2 * inch))

        # Metadata
        story.append(Paragraph(f"Date & Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Number of docs checked: {st.session_state.usage_count}", styles['Normal']))
        story.append(Paragraph(f"Billing summary: {st.session_state.usage_count * 10} INR", styles['Normal']))
        story.append(Spacer(1, 0.5 * inch))

        for i, (text, summary, filename) in enumerate(zip(original_texts, summaries, filenames)):
            story.append(Paragraph(f"--- Document {i+1}: {filename} ---", styles['h2']))
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph("Original Text:", styles['h3']))
            story.append(Paragraph(text.replace('\n', '<br />\n'), styles['BodyText']))
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph("Corrected Text (Summary):", styles['h3']))
            story.append(Paragraph(summary.replace('\n', '<br />\n'), styles['BodyText']))
            story.append(Spacer(1, 0.5 * inch))

        doc.build(story)

        with open(report_filename, "rb") as f:
            pdf_bytes = f.read()

        return report_filename, pdf_bytes


class View:
    """Handles the Streamlit UI components."""
    def __init__(self):
        # Initialize theme in session state
        if 'theme' not in st.session_state:
            st.session_state.theme = 'dark'
        self.apply_custom_styling(st.session_state.theme)

    def apply_custom_styling(self, theme='dark'):
        """Applies custom CSS to style the Streamlit app."""
        css = """
        <style>
        /* --- Base Theme & Typography --- */
        .stApp {
            background-color: #0b090a; /* Main dark background */
        }

        /* Set a clean, modern font stack for consistent typography */
        html, body, [class*="st-"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            color: #d3d3d3; /* Default light gray text */
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #161a1d; /* Slightly lighter dark shade for sidebar */
            border-right: 1px solid #660708; /* Subtle red accent border */
        }

        /* Main title styling */
        h1 {
            color: #f5f3f4; /* Off-white for main titles */
            font-weight: 600;
        }

        /* Subheader styling */
        h2, h3 {
            color: #ffffff; /* White for subheaders */
            font-weight: 500;
        }

        /* --- Interactive Elements: Buttons, Inputs, Uploaders --- */

        /* General button styling (st.button, st.download_button) */
        .stButton > button, .stDownloadButton > button {
            background-color: #a4161a; /* Primary red for actions */
            color: #ffffff;
            border: none; /* Remove default border for a cleaner look */
            border-radius: 20px; /* More rounded buttons */
            padding: 0.5rem 1rem;
            font-weight: 600;
            box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.25); /* Subtle shadow for depth */
            transition: background-color 0.2s, box-shadow 0.2s; /* Smooth transition on hover */
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background-color: #e5383b; /* Brighter red on hover */
            color: #ffffff;
            box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.3);
        }
        .stButton > button:focus, .stDownloadButton > button:focus {
            outline: none; /* Remove default outline */
            box-shadow: 0 0 0 3px #161a1d, 0 0 0 5px #e5383b; /* Create a 'glow' effect for clear focus */
        }

        /* Text input styling (st.text_input) */
        [data-testid="stTextInput"] input {
            background-color: #161a1d;
            color: #d3d3d3;
            border: 1px solid #b1a7a6;
            border-radius: 0.5rem;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #e5383b; /* Brighter red border on focus */
            box-shadow: 0 0 0 3px #660708; /* Add a glow for better visibility */
        }

        /* File uploader styling */
        [data-testid="stFileUploader"] {
            background-color: #161a1d;
            border: 2px dashed #b1a7a6;
            border-radius: 0.75rem;
        }
        [data-testid="stFileUploader"]:hover, [data-testid="stFileUploader"]:focus-within {
            border-color: #a4161a; /* Red accent on hover */
            box-shadow: 0 0 0 3px #161a1d, 0 0 0 5px #a4161a; /* Add glow on focus-within */
        }

        /* --- Containers & Messages --- */
        /* Style for st.info to highlight summaries */
        .stAlert[data-baseweb="alert"] {
            background-color: #161a1d;
            color: #d3d3d3;
            border-left: 5px solid #a4161a;
        }

        /* Style for st.error to make warnings more prominent */
        [data-testid="stNotification"] {
            background-color: #ba181b;
            color: #ffffff;
        }

        /* Container styling for individual document outputs */
        [data-testid="stVerticalBlockBorderWrapper"] {
             border: 1px solid #b1a7a6; /* Neutral gray border for subtle separation */
             border-radius: 0.75rem; /* Consistent rounded corners */
             padding: 1rem 1rem 1.5rem 1rem;
             background-color: #161a1d;
        }

        /* --- Responsive Design for Mobile --- */
        @media (max-width: 768px) {
            /* Reduce font sizes for better readability on small screens */
            h1 {
                font-size: 2rem;
            }
            h3 {
                font-size: 1.3rem;
            }
            /* Add vertical spacing between document containers when they stack */
            [data-testid="stVerticalBlockBorderWrapper"]:not(:last-child) {
                margin-bottom: 1.5rem;
            }
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit application."""
    view = View()
    doc_processor = DocumentProcessor()
    summarizer = Summarizer()
    exporter = Exporter()

    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0

    st.title("Multi-Document Summarizer")
    st.write("Upload 1 to 3 documents for summarization. Supported formats are `.docx`, `.txt`, `.pdf`, and `.zip` archives containing these files.")

    if not os.path.exists("reports"):
        os.makedirs("reports")

    uploaded_files = st.file_uploader("Upload documents or ZIP archives", type=["docx", "txt", "pdf", "zip"], accept_multiple_files=True)

    if uploaded_files:
        docs_to_process = doc_processor.process_uploaded_files(uploaded_files)
        num_docs = len(docs_to_process)

        if not 1 <= num_docs <= 3:
            st.error(f"Found {num_docs} documents. Please upload between 1 and 3 supported documents.")
            st.stop()

        st.session_state.usage_count += num_docs
        num_sentences = st.slider("Summary Length (number of sentences):", min_value=1, max_value=10, value=3)
        output_format = st.radio("Select Download Format:", ("Text (.txt)", "PDF"), horizontal=True)
        st.divider()

        cols = st.columns(num_docs)
        all_texts, all_summaries = [], []

        for i, doc_file in enumerate(docs_to_process):
            with cols[i]:
                with st.container(border=True):
                    st.subheader(f"Document {i+1}")
                    st.caption(doc_file.name)

                    if doc_file.name.endswith('.docx'):
                        raw_text = doc_processor.read_docx_file(doc_file)
                    elif doc_file.name.endswith('.txt'):
                        raw_text = doc_processor.read_text_file(doc_file)
                    elif doc_file.name.endswith('.pdf'):
                        raw_text = doc_processor.read_pdf_file(doc_file)
                    all_texts.append(raw_text)

                    if raw_text:
                        summary_text = summarizer.summarize_text(raw_text, num_sentences=num_sentences)
                        all_summaries.append(summary_text)
                        st.info(f"**Summary:**\n\n{summary_text}")

                        file_data, ext, mime = exporter.get_download_data(summary_text, output_format)
                        st.download_button(
                            label=f"Download as .{ext}",
                            data=file_data,
                            file_name=f"summary_{os.path.splitext(doc_file.name)[0]}.{ext}",
                            mime=mime,
                            key=f"download_button_{i}"
                        )
        st.divider()
        st.subheader("Billing Summary:")
        st.write(f"Docs Checked: {st.session_state.usage_count}")
        st.write(f"Total Bill: {st.session_state.usage_count * 10} INR")

        if st.button("Generate Report"):
            report_file = exporter.generate_report(all_texts, all_summaries, [f.name for f in docs_to_process])
            st.success(f"Report saved to {report_file}")

        # Create a placeholder for the PDF download button
        pdf_download_placeholder = st.empty()

        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                report_filename, pdf_bytes = exporter.generate_pdf_report(all_texts, all_summaries, [f.name for f in docs_to_process])
                pdf_download_placeholder.download_button(
                    label="Download Report as PDF",
                    data=pdf_bytes,
                    file_name=os.path.basename(report_filename),
                    mime="application/pdf"
                )
                st.success(f"PDF Report is ready for download. Also saved to {report_filename}")

if __name__ == "__main__":
    main()