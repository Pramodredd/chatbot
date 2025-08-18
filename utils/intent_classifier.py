from tools.llm import get_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from fastapi import HTTPException



def format_messages_for_summary(messages):
    """Formats a list of messages into a single string for the summary prompt."""
    formatted_lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"Assistant: {msg.content}")
    return "\n".join(formatted_lines)

async def classify_intent_service(payload):
    query = payload.query

    try:
        
        # Get the initialized LLM
        llm = get_llm()


        # Construct an enhanced, descriptive prompt
        final_prompt = (f"""
          You are a highly accurate customer query classification system.
Your job is to read a user’s message and assign exactly one category from the list below:

1. Business Information – General questions about the company, its products, services, pricing, policies, or general information not tied to a specific order.
2. Order Status – Requests about an existing or past order, delivery times, shipping status, tracking numbers, payment confirmation for a specific order.
3. Raise a Ticket – Reports of problems, technical issues, damaged products, incorrect orders, complaints, or anything needing customer support intervention.

Classification Rules:

Output only the category name (no extra text).

If a query contains multiple topics, choose the one most urgent or most directly actionable for the user.

If the query is vague, infer the most likely intent based on context and wording.

Handle typos, slang, and informal language without losing meaning.

Never answer the user’s question — only classify it.

Examples:
Input: "When is your Black Friday sale?" → Business Information
Input: "My laptop order #2345 hasn’t shipped yet" → Order Status
Input: "The product I got is missing a part" → Raise a Ticket
Input: "Yo, my package tracking ain’t moving" → Order Status
Input: "Payment didn’t go through but I got charged" → Raise a Ticket

Now classify this user query:
{query}"""
        ) 

        # 5. Query the LLM
        llm_response = await(llm.ainvoke(final_prompt))

        return {"intent": llm_response}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during LLM inference: {e}")
