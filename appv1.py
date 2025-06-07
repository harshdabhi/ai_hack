import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import os

# Set page config
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="👨‍⚕️",
    layout="wide"
)

# Custom CSS for chat interface
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.assistant {
    background-color: #475063;
}
.chat-message .content {
    width: 100%;
}
.chat-message .content p {
    margin: 0;
}
.emergency {
    background-color: #ff4444;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def create_medical_prompt():
    """Create the medical prompt template"""
    return ChatPromptTemplate.from_template("""
    You are an AI diagnostic assistant that extracts clinically relevant information from patient symptom descriptions using the provided medical documents and web search results.

    ## Core Role
    You are an AI diagnostic assistant that extracts clinically relevant information from patient symptom descriptions.

    ## Primary Diagnostic Protocol

    ### Input Processing
    Extract from patient text:
    - Symptom(s), duration, severity (1-10), modifying factors
    - Infer vital sign implications
    - Flag emotional urgency cues

    ### Standard Response Format

    1. Structured Symptom Summary
    ```
    Interpreted as: [Medical terminology translation]
    • Chief complaint: [Primary symptom]
    • Duration: [Timeframe]
    • Severity: [1-10 scale if provided]
    • Associated symptoms: [List]
    • Risk factors: [If mentioned]
    ```

    2. Likelihood Matrix
    ```
    │ Common (PCP-manageable)    │ Urgent (ER-needed)        │
    ├───────────────────────────┼──────────────────────────┤
    │ 1. [Most likely diagnosis] │ A. [Emergency condition]  │
    │ 2. [Second likely]         │ B. [Second emergency]     │
    │ 3. [Third consideration]   │                          │
    ```

    3. Next-Step Decision Tree
    ```
    « If [mild features] → "Try [conservative measure] for [timeframe]"
    « If [red flags] → "Seek care at [Urgent Care/ER] today"
    « Either way → "Recommend checking: [specific tests]"
    ```

    4. Sources of Information
    - Reference: [Document/Web sources used]

    ### Emergency Response Protocol
    🚨 Immediate triggers: Chest pain + cardiac risk, severe SOB, neurological symptoms, severe headache, active bleeding, suicidal ideation

    Context from documents:
    {context}
    
    Web search results:
    {web_search}
    
    Question: {input}
    
    Remember to:
    1. Always include appropriate medical disclaimers
    2. Recommend consulting healthcare professionals
    3. Flag any potential emergency situations
    4. Provide evidence-based information
    """)

def display_chat_message(role: str, content: str):
    """Display a chat message with proper styling"""
    with st.container():
        st.markdown(f"""
        <div class="chat-message {role}">
            <div class="content">
                <p><strong>{role.title()}:</strong> {content}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def check_emergency_symptoms(text: str) -> bool:
    """Check if the text contains emergency symptoms"""
    emergency_keywords = [
        "chest pain", "shortness of breath", "severe pain",
        "unconscious", "seizure", "stroke", "heart attack",
        "severe bleeding", "suicidal", "emergency"
    ]
    return any(keyword in text.lower() for keyword in emergency_keywords)

def initialize_components():
    """Initialize all components including LLM, embeddings, and tools"""
    try:
        # Set Tavily API key
        os.environ["TAVILY_API_KEY"] = "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"

        # Initialize Ollama LLM
        llm = Ollama(
            model="Gemma3:1b",
            base_url="http://host.docker.internal:11434",
            temperature=0.9
        )

        # Load and process the medical resource
        loader = PyPDFLoader('./resourcev1.pdf')
        docs = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        documents = text_splitter.split_documents(docs)
        
        # Create vector store
        db = Chroma.from_documents(
            documents,
            OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://host.docker.internal:11434"
            )
        )
        
        # Create retriever
        retriever = db.as_retriever()
        
        # Create Tavily tool
        tavily_tool = Tool(
            name="tavily_search",
            description="Search the web for general information using Tavily.",
            func=TavilySearch(max_results=1, topic="general")
        )
        
        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever,
            "medical_docs_search",
            "Search through medical documents for relevant information."
        )
        
        # Create tools list
        tools = [retriever_tool, tavily_tool]
        
        # Create prompt
        prompt = create_medical_prompt()
        
        # Initialize agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Create retrieval chain
        retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        
        return llm, retriever, agent, retrieval_chain
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None, None

def hybrid_query(input_text: str, llm, retriever, agent, retrieval_chain):
    """Hybrid query function that tries retrieval first, then falls back to agent"""
    response = retrieval_chain.invoke({"query": input_text})
    answer = response.get("result", "").strip()

    if not answer or "I don't know" in answer or "No relevant" in answer:
        st.info("Searching web for additional information...")
        return agent.run(input_text)
    else:
        return answer

# Streamlit UI
st.title("👨‍⚕️ Medical AI Assistant")
st.write("Your AI-powered medical diagnostic assistant")

# Initialize components
with st.spinner("Initializing medical AI assistant..."):
    llm, retriever, agent, retrieval_chain = initialize_components()
    if not all([llm, retriever, agent, retrieval_chain]):
        st.error("Failed to initialize medical AI assistant. Please check the logs for details.")
        st.stop()

# Create two columns for the layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📚 Medical Knowledge Base")
    st.write("Using Merck Manual and web search for comprehensive medical information")
    
    # Add medical disclaimer
    st.markdown("""
    ---
    ### ⚠️ Medical Disclaimer
    This AI assistant is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """)

with col2:
    st.subheader("💬 Symptom Assessment")
    # Display chat history
    for question, answer in st.session_state.chat_history:
        display_chat_message("user", question)
        display_chat_message("assistant", answer)
    
    # Chat input
    user_question = st.text_input("Describe your symptoms:", key="user_input")
    
    if user_question:
        # Check for emergency symptoms
        if check_emergency_symptoms(user_question):
            st.markdown("""
            <div class="emergency">
                ⚠️ EMERGENCY WARNING: Your symptoms may require immediate medical attention. 
                Please call emergency services (911) or visit the nearest emergency room.
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("Analyzing symptoms..."):
            response = hybrid_query(
                user_question,
                llm,
                retriever,
                agent,
                retrieval_chain
            )
            st.session_state.chat_history.append((user_question, response))
            st.rerun()