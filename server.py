from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv
import os

load_dotenv()

try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Translate the following into {language}:"),
        ("user", "{text}")
    ])

    parser = StrOutputParser()
    chain = prompt_template | model | parser

    app = FastAPI()
    add_routes(app, chain, path="/chain")

except Exception as e:
    import traceback
    traceback.print_exc()
    raise RuntimeError(f"Startup failed due to: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
