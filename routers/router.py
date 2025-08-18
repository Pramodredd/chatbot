
from fastapi import WebSocket , WebSocketDisconnect, APIRouter , HTTPException
from utils.intent_classifier import classify_intent_service
from schemas.schema import ConversationPayload , AIResponse
from tools.state_management import ChatState , app
import asyncio
import json
import traceback
router = APIRouter()
from langchain_core.messages import HumanMessage, AIMessage


@router.websocket("/ws/chat/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    """This endpoint handles the real-time, stateful chat conversation."""
    await websocket.accept()
    print(f"--- Client connected on thread_id: {thread_id} ---")

    # 1. Correctly configure the thread for the checkpointer
    config = {"configurable": {"thread_id": thread_id}}

    try:
        while True:
            # Wait for a message from the client
            user_message = await websocket.receive_text()
            print(f"Received from client ({thread_id}): {user_message}")

            input_for_graph = {"messages": [HumanMessage(content=user_message)]}
            
            # This will hold the complete AI response as it's being built
            full_ai_response = ""

            async for chunk in app.astream(input_for_graph, config=config):
                print(f"--- Raw Chunk from astream: {chunk} ---")

                # Check if a chunk from our agent node has arrived
                if "llm_agent" in chunk:
                    # Get the messages dictionary from the agent's output
                    agent_output = chunk.get("llm_agent", {})
                    if "messages" in agent_output:
                        new_messages = agent_output["messages"]
                        
                        if new_messages:
                            # This is the full string response from the LLM for this step
                            ai_response_str = new_messages[-1]
                            
                            # Prevent sending duplicate messages if the stream yields the same state again
                            if ai_response_str != full_ai_response:
                                full_ai_response = ai_response_str

                                message_to_send = ""
                                # --- START: New Conditional Logic ---
                                try:
                                    # Step 1: Attempt to parse the response as JSON
                                    data = json.loads(ai_response_str.strip())

                                    # Step 2: Check if it's a dictionary and has our confirmation message
                                    if isinstance(data, dict) and "confirmation_message" in data:
                                        # This is the JSON case: extract and send the user-facing message
                                        message_to_send = data["confirmation_message"]
                                    else:
                                        # It's valid JSON, but not the format we expect.
                                        # Fallback to sending the raw string just in case.
                                        message_to_send = ai_response_str

                                except (json.JSONDecodeError, TypeError):
                                    # Step 3: If parsing fails, it's just a regular chat message.
                                    # Send the whole string.
                                    message_to_send = ai_response_str
                                # --- END: New Conditional Logic ---

                                # Finally, if we have a message to send, send it to the client
                                if message_to_send:
                                    print(f"--- Sending to client: {message_to_send} ---")
                                    await websocket.send_text(message_to_send)
            #         # The messages key contains a list with the new AI response
            #         if "messages" in agent_output:
            #             new_messages = agent_output["messages"]
                        
            #             if new_messages:
            #                 # The response from the LLM is the last item in this list
            #                 ai_response_str = new_messages[-1] # This is the string content
                            
            #                 # Check if this is new content to avoid sending duplicates
            #                 if ai_response_str != full_ai_response:
            #                     full_ai_response = ai_response_str
            #                     print(f"--- Sending to client: {full_ai_response} ---")
            #                     await websocket.send_text(full_ai_response)

            # print(f"Finished streaming to client ({thread_id}): {full_ai_response}")

    except WebSocketDisconnect:
        print(f"--- Client disconnected from thread_id: {thread_id} ---")
    except Exception as e:
        print(f"An error occurred in thread {thread_id}:")
        traceback.print_exc()
        await websocket.send_text(f"Error: {e}")
        # It's good practice to close the connection on server-side error
        await websocket.close()

