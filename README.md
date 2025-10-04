# Chatbot

A real-time, stateful chatbot built with FastAPI and LangGraph that provides intelligent conversation handling with intent classification and memory management.

## Features

- **Real-time Communication**: WebSocket-based chat interface for instant messaging
- **Intent Classification**: Automatically categorizes user queries into:
  - Business Information
  - Order Status
  - Raise a Ticket
- **Stateful Conversations**: Maintains conversation history and context across sessions
- **Memory Management**: Automatic conversation summarization when message limit is reached
- **LLM Integration**: Uses Nebius AI (Mistral-Nemo-Instruct-2407) for natural language processing
- **Thread-based Sessions**: Each conversation maintains its own thread for isolated state management

## Architecture

The chatbot follows a modular architecture with the following components:

- **FastAPI Backend**: RESTful API with WebSocket support
- **LangGraph State Management**: Handles conversation flow and state transitions
- **Intent Classification**: Intelligent routing based on user intent
- **Memory Persistence**: SQLite-based conversation checkpointing
- **Modular Design**: Separated concerns with routers, schemas, tools, and utilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Pramodredd/chatbot.git
cd chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add:
```env
NEBIUS_API_KEY=your_nebius_api_key_here
```

## Usage

1. Start the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

2. Connect to the WebSocket endpoint:
```
ws://localhost:8000/ws/chat/{thread_id}
```

Replace `{thread_id}` with a unique identifier for each conversation session.

## API Endpoints

- **GET /**: Health check endpoint
- **WebSocket /ws/chat/{thread_id}**: Real-time chat interface

## Project Structure

```
chatbot/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── routers/
│   └── router.py          # WebSocket route handlers
├── schemas/
│   └── schema.py          # Pydantic data models
├── tools/
│   ├── llm.py            # LLM configuration and wrapper
│   └── state_management.py # LangGraph state management
└── utils/
    └── intent_classifier.py # Intent classification utilities
```

## Configuration

The chatbot uses several configurable parameters:

- **Message Limit**: Conversations are summarized after 5 messages to maintain performance
- **LLM Model**: Currently configured to use Mistral-Nemo-Instruct-2407 via Nebius AI
- **CORS**: Configured to allow all origins for development

## Dependencies

Key dependencies include:

- **FastAPI**: Web framework and WebSocket support
- **LangChain**: LLM integration and message handling
- **LangGraph**: State management and conversation flow
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server
- **OpenAI**: API client for LLM integration

## How It Works

1. **Connection**: Client connects via WebSocket with a unique thread ID
2. **Message Processing**: User messages are processed through the LangGraph state machine
3. **Intent Classification**: The system determines if the message is conversational or actionable
4. **Response Generation**: Based on intent, the system either:
   - Engages in conversation to clarify user needs
   - Classifies the query and provides structured output
5. **State Persistence**: Conversation state is maintained using checkpointing
6. **Memory Management**: Long conversations are automatically summarized

## Development

To modify the chatbot behavior:

1. **Intent Categories**: Update the classification categories in `tools/state_management.py`
2. **LLM Configuration**: Modify the LLM settings in `tools/llm.py`
3. **Conversation Flow**: Adjust the state graph logic in `tools/state_management.py`
4. **API Endpoints**: Add new routes in `routers/router.py`


// project end