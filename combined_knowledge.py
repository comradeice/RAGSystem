import typer
from typing import List,Optional
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.knowledge.combined import CombinedKnowledgeBase
from phi.knowledge.website import WebsiteKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader



import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Create a Combined Knowledge Base from PDF urls , website urls and local drive PDF
# PDF url links below are for demonstration and teaching purposes only

# Create first knowledge base for PDF urls with PG vector database for embeddings and retrieval 

url_pdf_knowledge_base = PDFUrlKnowledgeBase(

    urls=["https://s3.amazonaws.com/static.bitwiseinvestments.com/Research/Bitwise-The-Year-Ahead-10-Crypto-Predictions-for-2024.pdf",
          "https://www.flowtraders.com/media/i33f1ipp/crypto-etp-report.pdf",
          "https://25733253.fs1.hubspotusercontent-eu1.net/hubfs/25733253/Focus-Crypto%20Outlook%202024-09.01.24.pdf",
          ],
          vector_db=PgVector(table_name ="urlpdf_crypto_stock",db_url=db_url,),
)

# Create another knowledge base for website urls with PG vector database for embeddings and retrieval 

website_knowledge_base = WebsiteKnowledgeBase(
    urls=["https://coinmarketcap.com/",
          "https://finance.yahoo.com/"],
    # Table name: ai.website_documents
    vector_db=PgVector( table_name="website_documents",
        db_url=db_url),
)

# Create another knowledge base for PDF in local drive with PG vector database for embeddings and retrieval 

local_pdf_knowledge_base = PDFKnowledgeBase(
    path= r"C:\Users\User\OneDrive\Desktop\Phidata\ADTA5230.pdf",  # The exact local path in your system
    # Table name: ai.local_pdf_documents
    vector_db=PgVector(
        table_name="local_pdf_documents",
        db_url=db_url,
    ),
    reader=PDFReader(chunk=True),
)

# Combine the 3 different knowledge base for enhanced RAG System

knowledge_base = CombinedKnowledgeBase(
    sources=[
        url_pdf_knowledge_base,
        website_knowledge_base,
        local_pdf_knowledge_base,
    ],
    vector_db=PgVector(
        table_name="combined_documents",
        db_url=db_url,
    ),
)

# Create a storage for the vectors

storage = PgAssistantStorage(table_name="combined_assistant",db_url=db_url)

# Create a combined assistant function to handle the vector embeddings from combined knowledge base and method of storage

def combined_assistant(new : bool = False,user : str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids : List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run : {run_id}\n")
    else:
        print(f"Continue Running : {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__== "__main__":
    typer.run(combined_assistant)

