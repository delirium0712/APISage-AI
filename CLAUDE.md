# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APISage is an AI-powered OpenAPI 3.0 specification analyzer that provides comprehensive API documentation assessment, quality analysis, and compliance checking. It uses a two-stage architecture:
- **Stage 1**: Document evaluation and refinement (API spec analysis)
- **Stage 2**: Conversational RAG interface for API documentation Q&A

## Development Commands

### Starting the Application

```bash
# Full application startup (both backend and UI)
poetry run python -m api.main  # Start backend API on port 8080
poetry run python gradio_app.py  # Start Gradio UI on port 7860
```

### Testing

```bash
# Test endpoints (if test files exist)
poetry run pytest tests/  # Run test suite
```

### Dependency Management

```bash
# Install dependencies
poetry install

# Update dependencies
poetry update

# Add new dependency
poetry add <package-name>
```

### Code Quality

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Type checking
poetry run mypy .

# Linting
poetry run flake8 .
```

## Architecture Overview

### Core Components

1. **Backend API** (`/api/main.py`)
   - FastAPI application with multiple analysis endpoints
   - Key endpoints:
     - `/analyze` - Comprehensive API analysis using LLM
     - `/analyze-agentic` - Multi-agent collaborative analysis
     - `/analyze-with-evaluation` - Analysis with quality scoring
     - `/rag-query` - RAG endpoint for conversational Q&A (added at line 795)
   - Uses SimpleLLMManager with o1-mini model for reasoning tasks

2. **Enhanced RAG System Architecture**
   - **Hybrid RAG approach**: Both context-injection and vector-based retrieval
   - **Vector RAG**: ChromaDB with BGE-M3 embeddings for semantic search
   - **DeepEval Integration**: RAG Triad evaluation (Answer Relevancy, Faithfulness, Contextual Relevancy)
   - **Performance Monitoring**: Real-time metrics tracking and alerting
   - `/rag-query-v2` endpoint with enhanced evaluation capabilities

3. **Multi-Agent System** (`/infrastructure/agentic_orchestrator.py`)
   - Specialized agents for different analysis aspects:
     - Security Analyst
     - Documentation Reviewer
     - Performance Engineer
     - Integration Specialist
   - Agents collaborate to provide comprehensive analysis

4. **Gradio UI** (`/gradio_app.py`)
   - Five main analysis tabs:
     - Comprehensive Analysis
     - Multi-Agent Analysis
     - Evaluated Analysis
     - RAG Query Interface
   - AI Assistant (DeepEval-enhanced RAG)
   - Enhanced dark theme with visual elements
   - Real-time integration with backend APIs
   - DeepEval metrics display in responses

5. **LLM Manager** (`/infrastructure/llm_manager.py`)
   - Handles OpenAI API integration
   - Model configuration for GPT-4o, o1-mini, etc.
   - Manages temperature settings (o1 models don't support temperature)
   - Structured output support for agent responses

### Key File Locations

- **API Models**: Inline in `/api/main.py` (lines 67-128)
- **RAG Implementation**: `/api/main.py` (multiple endpoints)
- **Vector RAG**: `/infrastructure/vector_rag.py`
- **DeepEval Integration**: `/infrastructure/deepeval_enhanced.py`
- **Performance Monitor**: `/infrastructure/performance_monitor.py`
- **UI Application**: `/gradio_app.py`
- **Agent Orchestration**: `/infrastructure/agentic_orchestrator.py`
- **Configuration**: `/config/settings.py` and `.env`

## Configuration

### Environment Setup

```bash
# Copy example environment file
cp env.example .env

# Required environment variables:
OPENAI_API_KEY=your-api-key-here  # Critical for LLM features
```

### OpenAI API Key Configuration

The system requires a valid OpenAI API key for LLM features. Set it via:
1. Environment variable: `export OPENAI_API_KEY="sk-..."`
2. `.env` file: Edit `OPENAI_API_KEY=your-key`
3. UI: Use the API Configuration section in Gradio interface

## Common Issues and Solutions

### API Key Error (401)
If you see "Incorrect API key provided" errors, ensure OPENAI_API_KEY is set correctly.

### RAG Not Working
The RAG system requires:
1. Valid OpenAI API key
2. Uploaded OpenAPI specification file
3. Backend server running on port 8080

### Port Conflicts
Default ports:
- Backend API: 8080
- Gradio UI: 7860
Change in respective startup commands if needed.

## Development Workflow

### Adding New Features

1. **New API Endpoint**: Add to `/api/main.py` following existing patterns
2. **New Agent**: Create in `/infrastructure/` extending base classes
3. **UI Enhancement**: Modify `/gradio_app.py` with Gradio components
4. **New Analysis Type**: Extend evaluation system in `/evaluation/`

### Testing Changes

1. Restart servers after backend changes
2. For UI changes, refresh browser (Gradio auto-reloads)
3. Test with sample specs in `/test_social_api.json`
4. Use evaluation suite for comprehensive testing

## Key Technical Details

- **Python Version**: 3.10+ required (3.13 in venv)
- **Framework**: FastAPI backend, Gradio frontend
- **LLM Models**: OpenAI o1-mini (reasoning), GPT-4o (general)
- **Logging**: Structured logging with structlog
- **Package Management**: Poetry for dependencies
- **Async**: Full async/await support in API endpoints

## Project Status Notes

- **DeepEval Integration Complete**: Full RAG Triad evaluation working
- **Vector RAG Enhanced**: ChromaDB with BGE-M3 embeddings operational
- **Performance Monitoring Fixed**: Real-time metrics tracking without errors
- **UI Consolidated**: Single `gradio_app.py` with all features
- **Production Ready**: All systems tested and working with provided API key

## Recent Enhancements

### DeepEval RAG Evaluation
- RAG Triad metrics: Answer Relevancy, Faithfulness, Contextual Relevancy
- Self-explaining evaluation with detailed reasoning
- Multiple evaluation levels (BASIC, MODERATE, STRICT)
- Real-time confidence scoring

### Performance Monitoring
- Query latency tracking
- Success/failure rate monitoring
- Real-time alerting system
- Time-bucketed metrics (1min, 5min, 1hour)

### Vector RAG System
- ChromaDB vector store integration
- BGE-M3 multilingual embeddings
- Hybrid search with keyword + semantic
- Semantic caching for improved performance