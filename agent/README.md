# LLM Agent Service

A FastAPI-based service that provides various LLM-powered code verification capabilities.

## Features

- Solana smart contract verification
- TypeScript code verification
- Extensible agent workflow system
- RESTful API endpoints
- Async processing support
- Comprehensive error handling
- Logging and monitoring

## Project Structure

```
agent/
├── app/
│   ├── api/            # API routes and endpoints
│   ├── core/           # Core configuration and setup
│   ├── models/         # Data models and schemas
│   ├── services/       # Business logic and agent implementations
│   └── main.py         # Application entry point
├── tests/              # Test suite
└── requirements.txt    # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:
```env
OPENAI_API_KEY=your_api_key_here
CORS_ORIGINS=["http://localhost:3000"]
```

## Running the Service

Start the service with:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the service is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Available Endpoints

### Solana Verification
- POST `/api/v1/verify/solana`
  - Verifies Solana smart contract code
  - Request body: `{ "code": "your_code_here" }`

### TypeScript Verification
- POST `/api/v1/verify/typescript`
  - Verifies TypeScript code
  - Request body: `{ "code": "your_code_here" }`

## Development

### Adding New Agents

1. Create a new agent class in `app/services/`
2. Implement the required workflow steps
3. Add new routes in `app/api/routes/`
4. Update the main application to include new routes

### Testing

Run tests with:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 