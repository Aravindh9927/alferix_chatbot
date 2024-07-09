import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import streamlit as st

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=50):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embed_chunks(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def save_embeddings_and_chunks(embeddings, chunks, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings.cpu().numpy())
    with open(os.path.join(output_dir, 'chunks.txt'), 'w') as f:
        for chunk in chunks:
            f.write("%s\n" % chunk)

def load_embeddings_and_chunks(output_dir):
    embeddings = np.load(os.path.join(output_dir, 'embeddings.npy'))
    with open(os.path.join(output_dir, 'chunks.txt'), 'r') as f:
        chunks = f.readlines()
    return embeddings, chunks

def retrieve_relevant_chunks(query, embeddings, chunks, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=5):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], convert_to_tensor=True)
    cosine_scores = np.dot(embeddings, query_embedding.T).flatten()
    top_k_indices = np.argsort(cosine_scores)[-top_k:][::-1]
    return [(chunks[i], cosine_scores[i]) for i in top_k_indices]





import google.generativeai as genai

def get_answer_from_gpt3(query, relevant_chunks):
    openai.api_key = "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"  # Replace with your OpenAI API key
    
    # Filter chunks based on similarity score
    filtered_chunks = [chunk[0] for chunk in relevant_chunks if chunk[1] > 0.4]
    
    if not filtered_chunks:
        return "I don't know"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{query}\n\nContext:\n" + "\n".join(filtered_chunks)}
    ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200
    )

    if response and response.choices:
        answer = response.choices[0].message.content.strip()
        if "I don't know" in answer:
            return "I don't know"
        else:
            return answer
    else:
        return "No response from GPT-3.5"


pdf_path = "C:\Users\AravindV\Documents\Alferix\alferix_gpt.py" 
output_dir =  "C:\Users\AravindV\Documents\Alferix\chunk.txt" 

# Extract text from the PDF
text = extract_text_from_pdf(pdf_path)

# Chunk the text
chunks = chunk_text(text)

# Embed the chunks
embeddings = embed_chunks(chunks)

# Save embeddings and chunks
save_embeddings_and_chunks(embeddings, chunks, output_dir)

# Load embeddings and chunks
embeddings, chunks = load_embeddings_and_chunks(output_dir)




import streamlit as st

# Set up page configuration
st.set_page_config(page_title="Bot Q&A", page_icon=":robot:")



# Alternatively, use st.image function (uncomment the following line)
st.image("logo.jpg", width=200, caption="", use_column_width=False)

# Centered Header and introduction
st.markdown("""
<div style="text-align: center;">
    <h1>Chat Bot Q&A</h1>
    <p>Welcome to the Alferix Chat Bot Q&A using GPT-3.5. Please ask your HR Policy related questions below.</p>
</div>
""", unsafe_allow_html=True)

# User input
query = st.text_input("Enter your query:", key="query")

# Process query
if st.button("Submit"):
    if query:
        # Function calls to handle the query
        relevant_chunks = retrieve_relevant_chunks(query, embeddings, chunks)
        answer = get_answer_from_gpt3(query, relevant_chunks)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.error("Please enter a query to get a response.")

# Customize the appearance with minimal CSS
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
    }
    .stButton > button {
        width: 100%;
        color: white;
        background-color: #4CAF50;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stTextInput input {
        border-radius: 5px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)
