# C_First Chatbot API

A sophisticated financial chatbot API built with FastAPI, leveraging RAG (Retrieval-Augmented Generation) architecture with ChromaDB vector database and OpenAI's GPT models for intelligent, context-aware responses to finance-related queries.

## Features

- **Multi-language Support**: Handles queries in English, Hindi, and Hinglish
- **Semantic Search**: Uses vector embeddings for accurate question matching
- **RAG Architecture**: Combines retrieval-based and generative AI responses
- **Finance Domain Focus**: Automatically filters non-finance questions
- **Persistent Storage**: ChromaDB for efficient vector storage and retrieval
- **Smart Fallback**: GPT-4 integration for queries without exact matches
- **Security**: Folio number detection and protection

## Tech Stack

- **Framework**: FastAPI
- **Vector Database**: ChromaDB
- **AI Models**: OpenAI (text-embedding-3-small, GPT-4o-mini)
- **Language**: Python 3.8+

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- JSON data file (`json_data.json`)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd cfirst-chatbot
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Prepare your data**

Ensure `json_data.json` is in the root directory with the following structure:
```json
[
  {
    "id": "1",
    "question": {
      "english": "Question in English",
      "hinglish": "Question in Hinglish",
      "hindi": "Question in Hindi"
    },
    "answer": {
      "english": "Answer in English",
      "hinglish": "Answer in Hinglish",
      "hindi": "Answer in Hindi"
    }
  }
]
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
```http
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```http
GET /status
```
Returns API status and database statistics.

**Response:**
```json
{
  "status": "working",
  "message": "API is running smoothly",
  "database_count": 150
}
```

### 3. Chat Endpoint
```http
POST /chat
```

Send a question to the chatbot.

**Request Body:**
```json
{
  "question": "What is mutual fund?",
  "similarity_threshold": 0.7
}
```

**Response:**
```json
{
  "status": "success",
  "user_query": "What is mutual fund?",
  "bot_response": "A mutual fund is an investment vehicle...",
  "similarity_score": 0.8542
}
```

**Parameters:**
- `question` (required): User's query
- `similarity_threshold` (optional): Matching threshold (default: 0.7)

## How It Works

1. **Question Embedding**: User queries are converted to vector embeddings
2. **Semantic Search**: ChromaDB retrieves the most similar questions
3. **Similarity Check**: Matches above threshold return stored answers
4. **Finance Validation**: Low-similarity queries are checked for finance relevance
5. **GPT Fallback**: Finance-related queries get generated responses using context
6. **Multi-language**: Automatically returns answers in the query's language

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
cfirst-chatbot/
├── main.py                 # Main application file
├── json_data.json         # Q&A dataset
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── chroma_db/            # Vector database storage
└── README.md             # This file
```

## Configuration

### Similarity Threshold
Adjust the `similarity_threshold` parameter (0-1) to control matching strictness:
- Higher values (0.8-0.9): More strict matching
- Lower values (0.5-0.7): More lenient matching

### Models
The application uses:
- `text-embedding-3-small`: For generating embeddings
- `gpt-4o-mini`: For fallback responses

## Error Handling

The API handles various scenarios:
- Empty questions → 400 Bad Request
- Folio number queries → Protected response
- Non-finance questions → Filtered with appropriate message
- Server errors → 500 Internal Server Error with details

## Security Features

- **Folio Number Protection**: Automatically detects and blocks queries containing folio numbers (8+ digits)
- **Domain Restriction**: Only answers finance-related questions
- **Environment Variables**: Sensitive data stored in `.env`

## Performance

- **Persistent Storage**: ChromaDB maintains vectors across restarts
- **Batch Initialization**: Data loaded once at startup
- **Optimized Retrieval**: Returns top 2 results for context

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Running in Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Troubleshooting

**Issue**: ChromaDB not initializing
- Solution: Ensure write permissions for `./chroma_db` directory

**Issue**: OpenAI API errors
- Solution: Verify API key in `.env` file and check quota

**Issue**: Empty responses
- Solution: Check `json_data.json` format and encoding (UTF-8)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## Acknowledgments

- OpenAI for GPT models
- ChromaDB for vector database
- FastAPI for the excellent framework

---

**Note**: This chatbot is designed specifically for financial queries. For best results, ensure your questions are related to finance, investments, stock markets, or banking.
