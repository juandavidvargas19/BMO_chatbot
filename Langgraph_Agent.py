"""
Cached Components for PDF RAG Application

This module contains all the cached component initialization functions
for the PDF RAG application, including LLM, embeddings, vector store,
and RAG agent creation.
"""

import os
from typing import TypedDict, Annotated, Sequence
import streamlit as sl

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END


@sl.cache_resource
def create_rag_agent(_llm, _retriever):
    """
    Create RAG agent once per session.
    
    Args:
        _llm: Language model instance
        _retriever: Document retriever instance
        
    Returns:
        Compiled RAG agent graph
    """
    print("🤖 Creating RAG agent...")
    
    @tool
    def retriever_tool(query: str) -> str:
        """This tool searches and returns the information from a pdf document."""
        docs = _retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the pdf document."
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}")
        
        return "\n\n".join(results)
    
    tools = [retriever_tool]
    llm_with_tools = _llm.bind_tools(tools)
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
    
    def should_continue(state: AgentState):
        """Check if the last message contains tool calls."""
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
    
    system_prompt = """
    You are an intelligent AI assistant who answers questions about the pdf document loaded into your knowledge base.
    Use the retriever tool available to answer questions. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers (e.g. page, section). 
    """
    
    tools_dict = {tool.name: tool for tool in tools}
    
    def call_llm(state: AgentState) -> AgentState:
        """Function to call the LLM with the current state."""
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages
        message = llm_with_tools.invoke(messages)
        return {'messages': [message]}
    
    def take_action(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if not t['name'] in tools_dict:
                result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}
    
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True: "retriever_agent", False: END}
    )
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    
    rag_agent = graph.compile()
    
    # Save the graph (only once)
    try:
        graph_image = rag_agent.get_graph().draw_mermaid_png()
        with open("rag_agent_graph.png", "wb") as f:
            f.write(graph_image)
        print("✅ RAG agent graph saved as 'rag_agent_graph.png'")
    except Exception as e:
        print(f"⚠️ Could not save RAG agent graph: {e}")
    
    print("✅ RAG agent created successfully!")
    return rag_agent

@sl.cache_resource
def initialize_llm_and_embeddings():
    """
    Initialize LLM and embeddings once per session.
    
    Args:
        use_finetuned (bool): Whether to use fine-tuned model if available
    
    Returns:
        tuple: (llm, embeddings)
    """
    
    # Your fine-tuned model ID
    #finetuned_model_id = "ft:gpt-4o-2024-08-06:personal::Bew7X3c0"
    finetuned_model_id = None

    # Default model fallback
    default_model = "gpt-4o"
    
    # Determine which model to use
    if  finetuned_model_id:
        model_to_use = finetuned_model_id
        model_type = "Fine-tuned"
    else:
        model_to_use = default_model
        model_type = "Base"
    
    try:
        # Initialize LLM with selected model
        llm = ChatOpenAI(
            model=model_to_use, 
            temperature=0,
            # Optional: Add other parameters
            max_tokens=None,  # Let the model decide
            timeout=30,       # 30 second timeout
        )
        
        # Test the model with a simple query to ensure it works
        test_response = llm.invoke("Hello")
        
        print(f"✅ {model_type} model loaded successfully: {model_to_use}")
        
    except Exception as e:
        print(f"⚠️ Failed to load {model_type.lower()} model: {e}")
        
        # Fallback to default model if fine-tuned fails
        if  model_to_use != default_model:
            print(f"🔄 Falling back to base model: {default_model}")
            llm = ChatOpenAI(model=default_model, temperature=0)
            model_to_use = default_model
            model_type = "Base (Fallback)"
        else:
            raise e
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")    
    print("✅ LLM and embeddings initialized!")

    return llm, embeddings

@sl.cache_data
def load_and_process_pdf(pdf_path: str):
    """
    Load and process PDF once per session.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of processed document chunks
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
    """
    print(f"📄 Loading and processing PDF: {pdf_path}")
    
    # Safety measure
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pdf_loader = PyPDFLoader(pdf_path)
    
    try:
        pages = pdf_loader.load()
        print(f"✅ PDF loaded successfully with {len(pages)} pages")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        raise
    
    # Chunking Process
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    pages_split = text_splitter.split_documents(pages)
    print(f"✅ PDF processed into {len(pages_split)} chunks")
    return pages_split


@sl.cache_resource
def create_vector_store(_pages_split, _embeddings):
    """
    Create vector store once per session.
    
    Args:
        _pages_split: Processed document chunks
        _embeddings: Embeddings model instance
        
    Returns:
        ChromaDB vector store instance
        
    Raises:
        Exception: If vector store creation fails
    """
    print("🗂️ Creating vector store...")
    
    persist_directory = os.getcwd()
    collection_name = "pdf_document"
    
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    try:
        vectorstore = Chroma.from_documents(
            documents=_pages_split,
            embedding=_embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print("✅ ChromaDB vector store created successfully!")
        return vectorstore
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {str(e)}")
        raise
