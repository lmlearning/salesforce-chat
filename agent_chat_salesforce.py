import logging
import sys
from langchain.agents import Tool
from langchain.schema import Memory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index import GPTVectorStoreIndex
import os
from llama_index import StorageContext, load_index_from_storage

os.environ["openai_api_key"] = "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"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir=".\\storage")

# load index
index = load_index_from_storage(storage_context)

tools = [
    Tool(
        name = "LlamaIndex",
        func=lambda q: str(index.as_query_engine().query(q)),
        description="useful for when you want to answer questions about Salesforce, MuleSoft, Tableau, Quip, CRM, MarTech, or tech in general.",
        return_direct=True
    ),
]

# set Logging to DEBUG for more detailed outputs
memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatOpenAI(temperature=0)
agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)
while True:
    in_val = input()
    print(agent_executor.run(input=in_val))
