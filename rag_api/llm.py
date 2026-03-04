from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import markdown

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

QUERY_MODEL: str = "openai/gpt-oss-20b"
SAFEGUARD_MODEL: str = "openai/gpt-oss-safeguard-20b"

_llm = ChatGroq(
    model=QUERY_MODEL,
    temperature=0.2,
)

def explain_chunks(chunks: list[dict], user_query: str, format_type="html") -> str:
    text_block = "\n\n".join(
        f"Chunk {i+1}:\n{c.get('text','')}"
        for i, c in enumerate(chunks)
    )

    prompt = f"""
    USER QUERY:
    \"\"\"{user_query}\"\"\"

    TASK:
    Provide a clear, accurate, human-readable explanation that answers the user query
    using ONLY the retrieved content below. Do not invent information.

    RETRIEVED CONTENT:
    {text_block}

    EXPLANATION:
    """
    response = _llm.invoke(prompt)
    if format_type == "html":
        return render_markdown(str(response.content))
    else: # otherwise return raw llm markdown 
        return str(response.content) # casts "content" as str to satisfy pylance, even though it already is a str.

def fix_newlines(text: str) -> str:
    return text.replace("\n", "")
    
def render_markdown(text: str) -> str:
    """Render markdown text to HTML safely."""
    return markdown.markdown(
        # fix_newlines(text),
        text,
        extensions=['tables', 'fenced_code', 'codehilite']
    )