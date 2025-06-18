from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

# === Konfiguracja ===
load_dotenv()
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

loader = DirectoryLoader("planets", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=50, chunk_overlap=0
)
split_docs = []
for doc in documents:
    for chunk in text_splitter.split_text(doc.page_content):
        split_docs.append(Document(page_content=chunk))
db = Chroma.from_documents(split_docs, embedding=embeddings)

# === Tools ===
@tool
def planet_distance_sun(planet_name: str) -> str:
    """This tool return planets approximate distance from the Sun in Astronomical Units (AU)"""
    if "earth" in planet_name.lower():
        return "Earth is approximately 1 AU from the Sun."
    elif "mars" in planet_name.lower():
        return "Mars is approximately 1.5 AU from the Sun."
    elif "jupiter" in planet_name.lower():
        return "Jupiter is approximately 5.2 AU from the Sun."
    elif "pluto" in planet_name.lower():
        return "Pluto is approximately 39.5 AU from the Sun."
    else:
        return f"Information about the distance of {planet_name} from the Sun is not available in this tool."

@tool
def planet_revolution_period(planet_name: str) -> str:
    """This tool returns planets approximate revolution periods around the Sun in Earth years"""
    if "earth" in planet_name.lower():
        return "Earth takes approximately 1 Earth year to revolve around the Sun."
    elif "mars" in planet_name.lower():
        return "Mars takes approximately 1.88 Earth years to revolve around the Sun."
    elif "jupiter" in planet_name.lower():
        return "Jupiter takes approximately 11.86 Earth years to revolve around the Sun."
    elif "pluto" in planet_name.lower():
        return "Pluto takes approximately 248 Earth years to revolve around the Sun."
    else:
        return f"Information about the revolution period of {planet_name} is not available in this tool."

@tool
def planet_general_info(planet_name: str) -> str:
    """This toll handles general planet queries that are not about the planet's distance from the Sun or its revolution period."""
    docs = db.similarity_search(planet_name)
    if docs:
        return docs[0].page_content
    else:
        return f"Additional information for {planet_name} is not available in this tool."

tools_list = [
    planet_distance_sun,
    planet_revolution_period,
    planet_general_info
    ]
function_implementations = {
    "PlanetDistanceSun": planet_distance_sun,
    "PlanetRevolutionPeriod": planet_revolution_period,
    "PlanetGeneralInfo": planet_general_info,
}

# === Bind tools ===
model_with_tools = llm.bind_tools(tools_list)

# === Łańcuch 1: formatowanie zapytania ===
prompt_template = PromptTemplate.from_template(
    "You are a helpful assistant who answers questions users may have. You are asked about: {planet}."
    "Format the output for readability to make it easy for users to extract useful information at a glance"
)
#format_chain = prompt_template | llm

# === Łańcuch 2: model z narzędziami + wykonanie funkcji ===
def execute_tools(ai_message):
    for call in ai_message.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        tool = function_implementations.get(tool_name)
        if tool:
            result = tool.run(tool_args)
            return result
    return None

tool_executor_chain = RunnableLambda(execute_tools)

# === Finalny łańcuch ===
final_chain = prompt_template | model_with_tools | tool_executor_chain

# === Wykonanie ===
query = input()
result = final_chain.invoke({"planet": query})
print(result)
print()
print(final_chain)
