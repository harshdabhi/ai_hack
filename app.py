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
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
if "components_initialized" not in st.session_state:
    st.session_state.components_initialized = False
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "current_response" not in st.session_state:
    st.session_state.current_response = None

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

    ### Standard Response Format - YOU MUST FOLLOW THIS EXACT FORMAT:

    1. Structured Symptom Summary
    ```
    Interpreted as: [Medical terminology translation]
    • Chief complaint: [Primary symptom]
    • Duration: [Timeframe]
    • Severity: [1-10 scale if provided]
    • Associated symptoms: [List]
    • Risk factors: [If mentioned]
    • Vital signs implications: [If relevant]
    • Emotional/psychological factors: [If present]
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

    4. Detailed Analysis
    ```
    • Pathophysiology: [Brief explanation of the condition]
    • Risk factors: [Detailed list of risk factors]
    • Complications: [Potential complications if untreated]
    • Treatment options: [Available treatments and their effectiveness]
    • Prevention: [Preventive measures if applicable]
    ```

    5. Sources of Information
    - Reference: [Document/Web sources used]
    - Evidence level: [Strength of evidence]

    ### Emergency Response Protocol
    🚨 Immediate triggers: Chest pain + cardiac risk, severe SOB, neurological symptoms, severe headache, active bleeding, suicidal ideation

    Context from documents:
    {context}
    
    Web search results:
    {web_search}
    
    Question: {input}
    
    IMPORTANT INSTRUCTIONS:
    1. You MUST follow the exact format above
    2. You MUST include ALL sections in your response
    3. You MUST use the exact formatting shown
    4. You MUST provide detailed information in each section
    5. You MUST include appropriate medical disclaimers
    6. You MUST recommend consulting healthcare professionals
    7. You MUST flag any potential emergency situations
    8. You MUST provide evidence-based information
    9. You MUST be thorough and detailed in your response
    10. You MUST include all relevant medical terminology
    11. You MUST explain complex terms in simple language
    12. You MUST provide specific recommendations when possible
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
        logger.info("Starting component initialization...")
        start_time = datetime.now()

        # Set Tavily API key
        os.environ["TAVILY_API_KEY"] = "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"
        logger.info("Tavily API key set")

        # Initialize Ollama LLM
        logger.info("Initializing Ollama LLM...")
        llm = Ollama(
            model="Gemma3:1b",
            base_url="http://host.docker.internal:11434",
            temperature=0.7,  # Lower temperature for m
        )
        logger.info("Ollama LLM initialized")

        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://host.docker.internal:11434"
        )

        # Check if database exists
        persist_directory = "./chroma_db"
        if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
            logger.info("Using existing vector store from disk")
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            logger.info("Loaded existing vector store")
        else:
            logger.info("Creating new vector store...")
            # Load and process the medical resource
            loader = PyPDFLoader('./resourcev1.pdf')
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents from PDF")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            documents = text_splitter.split_documents(docs)
            logger.info(f"Split into {len(documents)} chunks")
            
            # Create and persist vector store
            db = Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=persist_directory
            )
            db.persist()
            logger.info("Created and persisted new vector store")
        
        # Create retriever
        retriever = db.as_retriever()
        
        # Create Tavily tool
        logger.info("Creating Tavily tool...")
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
        logger.info("Initializing agent...")
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Create retrieval chain
        logger.info("Creating retrieval chain...")
        retrieval_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        
        # Store components in session state
        st.session_state.llm = llm
        st.session_state.retriever = retriever
        st.session_state.agent = agent
        st.session_state.retrieval_chain = retrieval_chain
        st.session_state.components_initialized = True
        
        end_time = datetime.now()
        logger.info(f"Component initialization completed in {(end_time - start_time).total_seconds():.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        return False

def hybrid_query(input_text: str):
    """Hybrid query function that tries retrieval first, then falls back to agent"""
    try:
        logger.info(f"Processing query: {input_text}")
        start_time = datetime.now()

        response = st.session_state.retrieval_chain.invoke({"query": input_text})
        answer = response.get("result", "").strip()

        if not answer or "I don't know" in answer or "No relevant" in answer:
            logger.info("No relevant local information found, falling back to web search...")
            st.info("Searching web for additional information...")
            answer = st.session_state.agent.run(input_text)

        end_time = datetime.now()
        logger.info(f"Query processed in {(end_time - start_time).total_seconds():.2f} seconds")
        return answer
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return "I apologize, but I encountered an error while processing your query. Please try again."

# Streamlit UI
st.title("👨‍⚕️ Medical AI Assistant")
st.write("Your AI-powered medical diagnostic assistant")

# Initialize components only once
if not st.session_state.components_initialized:
    with st.spinner("Initializing medical AI assistant..."):
        if not initialize_components():
            st.error("Failed to initialize medical AI assistant. Please check the logs for details.")
            st.stop()
        st.success("Medical AI assistant initialized successfully!")

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
    
    if user_question and user_question != st.session_state.last_question:
        st.session_state.last_question = user_question
        
        # Check for emergency symptoms
        if check_emergency_symptoms(user_question):
            st.markdown("""
            <div class="emergency">
                ⚠️ EMERGENCY WARNING: Your symptoms may require immediate medical attention. 
                Please call emergency services (911) or visit the nearest emergency room.
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("Analyzing symptoms..."):
            response = hybrid_query(user_question)
            st.session_state.current_response = response
            st.session_state.chat_history.append((user_question, response))
            display_chat_message("assistant", response)