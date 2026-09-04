from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

MURPHY_FILENAMES = {"NATURES_MATERIA_MEDICA.txt", "METAREPERTORY.txt"}
DISABLED_FILENAMES = {"BOENNINGHAUSEN_POCKET_BOOK.txt"}  # temporarily excluded, indexed but not served

class MaterialMedicaRAG:
    def __init__(self, pdf_paths, db_path: str = "../data/chroma_db"):
        self.pdf_paths = pdf_paths if isinstance(pdf_paths, list) else [pdf_paths]
        self.db_path = db_path
        self.vectorstore = None

        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        print("✅ Embedding model ready")

    def index(self):
        all_chunks = []
        for pdf_path in self.pdf_paths:
            print(f"📖 Loading: {pdf_path}")
            try:
                path = Path(pdf_path)
                if path.suffix == '.txt':
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    docs = [Document(page_content=text, metadata={"source": str(path)})]
                else:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                
                print(f"   loaded")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80,
                    separators=["\n\n", "\n", ".", " "],
                )
                chunks = splitter.split_documents(docs)
                print(f"   {len(chunks)} chunks created")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️  Error loading {pdf_path}: {e}")

        print(f"🔍 Building vector database with {len(all_chunks)} total chunks...")
        self.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path,
        )
        print("✅ All books indexed and ready")

    def load(self):
        print("📂 Loading existing vector database...")
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
        )
        count = self.vectorstore._collection.count()
        print(f"✅ Loaded {count} chunks from database")

    def search(self, query: str, k: int = 4, allow_murphy: bool = False) -> str:
        if not self.vectorstore:
            return ""
        fetch_k = k * 4  # overfetch so filtering doesn't starve real results
        docs = self.vectorstore.similarity_search(query, k=fetch_k)
        docs = [d for d in docs if Path(d.metadata.get("source", "")).name not in DISABLED_FILENAMES]
        if not allow_murphy:
            docs = [d for d in docs if Path(d.metadata.get("source", "")).name not in MURPHY_FILENAMES]
        docs = docs[:k]
        passages = [doc.page_content.strip() for doc in docs]
        return "\n\n".join(passages)