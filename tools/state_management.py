
import json
from typing import List, TypedDict , Annotated
from dotenv import load_dotenv


from langchain_core.messages import AIMessage, BaseMessage, HumanMessage , SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.llm import get_llm
from utils.intent_classifier import format_messages_for_summary
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()
llm = get_llm()
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage] , add_messages]    
    query : str
    res : List[BaseMessage]
    summary : str


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            # "system",
            # """You are a helpful AI assistant. Your task is to analyze the user's query and classify it.
            
            # Based on the conversation history and the user's latest query, classify it into ONE of the following categories:
            # - Business Information
            # - Raise a Ticket
            # - Order Status

            # If the user's query is a clear, actionable request that fits one of these categories, you MUST respond ONLY with a valid JSON object of the following format:
            # {{"category": "<category_name>", "query": "<the user's query>"}}

            # If the user's query is conversational (e.g., "hello", "how are you?"), ambiguous, or does not fit a category, engage in a friendly conversation to clarify their needs. DO NOT output JSON in this case. {{"classification": {{"category": "<category_name>", "query": "<the user's query>"},
    #  "confirmation_message": "<Confirmation message to users query being handled.>"}}
            "system",
            """You are a friendly, helpful AI assistant whose main task is to understand the user's intent through natural, engaging conversation before classifying it.  

Strictly follow**Carefully read both the conversation history and the user's most recent message. Your classification categories are:
- Business Information
- Raise a Ticket
- Order Status**

Process:
1. Only classify the user's request if it is **clear, specific, and unambiguous** enough to confidently match exactly ONE category from 
- Business Information
- Raise a Ticket
- Order Status 
   - Respond **only** with a valid JSON object in this exact format:
                {{
                  "classification": {{
                    "category": "<category_name>",
                    "query": "<a summary of the user's query>"
                  }},
                  "confirmation_message": "<A friendly confirmation message to send to the user>"
                }}
                


2. If the request is **ambiguous, vague, or conversational** (e.g., greetings, small talk, unclear intent), do **not** guess or make assumptions.  
   - Respond in a warm, human-like tone to build rapport.  
   - Ask open-ended questions to clarify what the user needs.  
   - Only classify once the user’s request is clear.
   - You answer with a statement like i take messages about orders , business information and problems with products or orders.

Tone guidelines:
- Be approachable, professional, and concise.  
- Use conversational language, not robotic phrasing.  
- Avoid hallucinating details, making assumptions, or forcing a classification.

---
**CRITICAL OUTPUT RULE:**
- You are the AI assistant. Your response MUST begin directly with your message.
- You MUST NOT under any circumstances prefix your response with speaker labels like "Human:", "User:", "AI:", or "Assistant:".
---


"""
,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

summarization_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a conversation summarization assistant. Your task is to create a concise summary of the following conversation.
The summary should be from the perspective of the AI assistant and must capture all key information, context, user details (like names or order numbers), and any unresolved questions.
This summary will be used as a memory for the AI to continue the conversation without the full history. It needs to be dense with information."""
        ),
        ("user", "Please summarize this conversation:\n\n{conversation_history}"),
    ]
)

summarizer_chain = summarization_prompt | llm | StrOutputParser()

def call_llm_node(state: ChatState):
    print("---LOG: Calling LLM---")

    chain = prompt_template | llm
    
    response = chain.invoke({"messages": state["messages"]})
    
    return {"messages": [response]}




def should_continue_json_check(state: ChatState) -> str:
    """
    Checks the content of the last message in the state.
    This check is robust against object serialization by the checkpointer.
    """
    last_message = state['messages'][-1]
    content_to_check = ""

    if hasattr(last_message, 'content'):
        content_to_check = last_message.content
    elif isinstance(last_message, str):
        content_to_check = last_message
    else:
        return "continue"
    
    try:
        # We strip any whitespace or newlines that might surround the JSON
        data = json.loads(content_to_check.strip())
        
        # The actual check for the JSON structure
        if isinstance(data, dict) and "category" in data and "query" in data:
            print(f"--- Valid JSON command detected by graph. Ending. ---")
            return "end"  # This will now correctly end the graph
        else:
            # It's JSON, but not the format we want
            return "continue"
            
    except (json.JSONDecodeError, TypeError):
        # The content was not a valid JSON string, so we continue the conversation
        return "continue"


def human_in_the_loop_node(state: ChatState):
    """
    This node is a placeholder. The graph will pause *before* executing it.
    It doesn't need to do anything itself. Its name is its function.
    """
    pass


MESSAGE_LIMIT = 5

def summarize_node(state: ChatState) -> dict:
    """If the conversation is long, summarize the older messages."""
    messages = state["messages"]
    
    if len(messages) <= MESSAGE_LIMIT:
        return {} 

    print("--- History limit reached. Summarizing older messages... ---")

    messages_to_summarize = messages[:-2]
    
    conversation_str = format_messages_for_summary(messages_to_summarize)

    new_summary_text = summarizer_chain.invoke(
        {"conversation_history": conversation_str}
    )
    print(f"--- Generated Summary: {new_summary_text} ---")

    summary_message = SystemMessage(content=new_summary_text)

    updated_messages = [summary_message] + messages[-2:]

    return {
        "messages": updated_messages,
        "summary": new_summary_text 
    }


# LangGraph setup
memory = MemorySaver()
workflow = StateGraph(ChatState)

workflow.add_node("llm_agent", call_llm_node)
workflow.add_node("summarizer", summarize_node)
workflow.add_node("human_in_the_loop", human_in_the_loop_node)
workflow.set_entry_point("llm_agent")
workflow.add_edge("llm_agent", "summarizer")
workflow.add_conditional_edges(
    "summarizer",
    should_continue_json_check,
    {
        "continue": "human_in_the_loop", # If we continue, go to the pause node
        "end": END                      # If we end, exit the graph
    }
)


app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_in_the_loop"]
)
print("--- LangGraph App with MemorySaver Compiled Successfully ---")

