import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = Flask(__name__)

API_KEY = "VOTRE_API_KEY_GEMINI"
os.environ["GOOGLE_API_KEY"] = API_KEY

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
VECTOR_STORE_FOLDER = 'vector_stores'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VECTOR_STORE_FOLDER'] = VECTOR_STORE_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_process_pdf(file_path: str) -> List[dict]:
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_text(page.page_content)
        for chunk in page_chunks:
            chunks.append({
                "content": chunk,
                "page": page.metadata['page'] + 1
            })
    
    return chunks

def create_vector_store(chunks: List[dict], filename: str) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"page": chunk["page"]} for chunk in chunks]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Save the vector store
    vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], f"{filename}.faiss")
    vectorstore.save_local(vector_store_path)
    
    return vectorstore

def load_vector_store(filename: str) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], f"{filename}.faiss")
    try:
        return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        if "allow_dangerous_deserialization" in str(e):
            # If the error is about deserialization, try loading without the parameter
            return FAISS.load_local(vector_store_path, embeddings)
        else:
            # If it's a different error, re-raise it
            raise

def create_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2, google_api_key=API_KEY)
    
    template = """You are a legal assistant. Use the following pieces of 
    context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Always include references to the relevant law articles you used to formulate your answer.

    Context: {context}

    Question: {question}

    Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

@app.route('/')
def index():
    books = [f.replace('.faiss', '') for f in os.listdir(app.config['VECTOR_STORE_FOLDER']) if f.endswith('.faiss')]
    return render_template('index.html', books=books)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        chunks = load_and_process_pdf(file_path)
        create_vector_store(chunks, filename.rsplit('.', 1)[0])
        
        return jsonify({'message': 'File uploaded and processed successfully'}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    book = data.get('book')
    question = data.get('question')
    
    if not book or not question:
        return jsonify({'error': 'Missing book or question'}), 400
    
    vectorstore = load_vector_store(book)
    qa_chain = create_qa_chain(vectorstore)
    
    result = qa_chain({"query": question})
    answer = result['result']
    source_documents = result['source_documents']
    
    sources = []
    for doc in source_documents:
        sources.append({
            'content': doc.page_content,
            'page': doc.metadata.get('page', 'Unknown')
        })
    
    return jsonify({
        'answer': answer,
        'sources': sources
    })

if __name__ == '__main__':
    app.run(debug=True)