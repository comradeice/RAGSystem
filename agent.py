from phi.agent import Agent
from combined_knowledge import knowledge_base
import typer
from typing import List,Optional
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
storage = PgAssistantStorage(table_name="combined_assistant",db_url=db_url)

agent = Agent(
    knowledge=knowledge_base,
    search_knowledge=True,
)
agent.knowledge.load(recreate=False)


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