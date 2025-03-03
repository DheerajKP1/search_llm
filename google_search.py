import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import chromadb
import uuid
load_dotenv()
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

# Set up LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="deepseek-r1-distill-llama-70b"
)

os.environ["GOOGLE_CSE_ID"] =os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

search = GoogleSearchAPIWrapper()

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="google_search",
    description="Search Google for recent results and give consize in 2-3 lines answer.",
    func=search.run,
)
# bound_model = model.bind_tools(tools)


st.title("AI-Powered Question Answering Chatbot") 
st.write("This chatbot is designed to find the comptetiors of a company in their domain")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your company and find your competitors")

if query:
    with st.spinner("Thinking..."):
        # Extract the result
        question  =query
        ans=tool.run(question)
        response=llm.invoke(f"This is recent update I have found through google search Information: {ans}, so just make it to the point for this question:{question}")
        cleaned_response= [line for line in re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).split("\n") if line.strip()]
        cleaned_response_str = "\n".join(cleaned_response)

        follow_up_prompt = f'''just Generate 2-3 follow-up questions related to query : {query} please put it like would you like know more about this then follow up question but do not add any extra information.'''
        follow_up_response = llm.invoke(follow_up_prompt)
        follow_up_questions = follow_up_response
        follow_up_questions = re.sub(r"<think>.*?</think>", "", follow_up_questions.content, flags=re.DOTALL).split("\n")

        # Update chat history
        st.session_state.chat_history.append(HumanMessage(query))
        st.session_state.chat_history.append(AIMessage(cleaned_response_str))
        st.write("### Answer:")
        st.write(cleaned_response_str)
        if follow_up_questions:
            st.write("\n💡 **Follow-up Questions:**")
            for q in follow_up_questions:
                if q.strip():
                    st.write("-", q.strip())

# Display past conversation
if st.session_state.chat_history:
    st.write("\n\n### Chat History:")
    for message in st.session_state.chat_history:
        role = "👤 You:" if isinstance(message, HumanMessage) else "🤖 Bot:"
        st.write(f"{role} {message.content}")
