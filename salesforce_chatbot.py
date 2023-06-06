import streamlit as st
from langchain.agents import Tool
from langchain.schema import Memory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index import GPTVectorStoreIndex
import os
from llama_index import StorageContext, load_index_from_storage
from streamlit_chat import message


st.set_page_config(
    page_title="Salesforce Chat - Demo",
    page_icon=":robot:"
)

@st.cache_resource
def get_index_and_agent():
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="storage")

    # load index
    index = load_index_from_storage(storage_context)

    tools = [
        Tool(
            name="LlamaIndex",
            func=lambda q: str(index.as_query_engine().query(q)),
            description="useful for when you want to answer questions about Salesforce, MuleSoft, Tableau, Quip, CRM, MarTech, or tech in general.",
            return_direct=True
        ),
    ]

    # set Logging to DEBUG for more detailed outputs
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(temperature=0.5)
    agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)
    return index, agent_executor

index,agent_executor = get_index_and_agent()


st.header("Salesforce Chat - Demo")
st.markdown("[Github](https://github.com/lmlearning/salesforce-chat)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = agent_executor.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    input_text = ""

if st.session_state['generated']:
     for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')