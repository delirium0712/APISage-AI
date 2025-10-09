"""
APISage Complete Application - Stage 1 (Evaluation) + Stage 2 (Assistant)
Combines analysis functionality with RAG assistant in tabbed interface
"""

import gradio as gr
import json
import yaml
import os
import httpx
import asyncio
import logging
import sys
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("apisage_complete.log"),
    ],
)

logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "60"))
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "120"))

# Global state
current_spec = None

# Helper functions for consistent error handling
def handle_api_error(status_code: int, operation: str = "operation") -> str:
    """Provide detailed, actionable error messages for API responses"""
    error_messages = {
        400: f"❌ **Invalid Request** for {operation}\n\n**Cause**: The request format is incorrect\n**Solution**: \n• Check that your API specification is valid JSON/YAML\n• Ensure all required fields are provided\n• Verify the document structure matches OpenAPI 3.0 format",

        401: f"❌ **Authentication Failed** for {operation}\n\n**Cause**: Invalid or missing OpenAI API key\n**Solution**: \n• Ensure your OpenAI API key is properly set in the .env file\n• Get your API key from https://platform.openai.com/api-keys\n• Verify the key starts with 'sk-' and has sufficient credits\n• Restart the backend server after updating the .env file",

        403: f"❌ **Access Denied** for {operation}\n\n**Cause**: Insufficient permissions or quota exceeded\n**Solution**: \n• Check your OpenAI account quota and billing\n• Verify your API key has the necessary permissions\n• Contact OpenAI support if the issue persists",

        404: f"❌ **Service Not Found** for {operation}\n\n**Cause**: Backend server is not running or unreachable\n**Solution**: \n• Ensure the backend server is running on port 8080\n• Try refreshing the page and waiting 30 seconds\n• Check the server logs for startup errors",

        429: f"❌ **Rate Limit Exceeded** for {operation}\n\n**Cause**: Too many requests sent to the API\n**Solution**: \n• Wait 60 seconds before trying again\n• Consider upgrading your OpenAI plan for higher limits\n• Try using a smaller document or shorter questions",

        500: f"❌ **Server Error** during {operation}\n\n**Cause**: Internal server processing error\n**Solution**: \n• Try again in a few moments\n• If using a large document, try breaking it into smaller sections\n• Check that your OpenAPI spec is properly formatted\n• Contact support if the error persists",

        503: f"❌ **Service Unavailable** for {operation}\n\n**Cause**: OpenAI API key not configured or service down\n**Solution**: \n• Set your OpenAI API key in the Configuration tab\n• Verify your API key is active and has credits\n• Check OpenAI's status page: https://status.openai.com",

        504: f"❌ **Request Timeout** for {operation}\n\n**Cause**: Request took too long to process\n**Solution**: \n• Try again with a smaller document\n• Ask shorter, more specific questions\n• Check your internet connection\n• Consider breaking complex queries into smaller parts"
    }

    default_error = f"❌ **Unexpected Error** during {operation} (Status: {status_code})\n\n**Cause**: Unknown server response\n**Solution**: \n• Try refreshing the page and retrying\n• Check your internet connection\n• Contact support if the error persists\n• Include this status code: {status_code}"

    return error_messages.get(status_code, default_error)

def log_error_with_context(error: Exception, context: str, user_message: str = None) -> str:
    """Log errors with context and return user-friendly message"""
    logger.error(f"{context}: {str(error)}", exc_info=True)
    return user_message or f"❌ {context} failed: {str(error)}"

# 2025 Modern UI/UX Design with Glassmorphism and Enhanced Accessibility
CUSTOM_CSS = """
/* 2025 Design System with Glassmorphism and Modern Patterns */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --neon-accent: #00f5ff;
    --success-glow: #10b981;
    --warning-glow: #f59e0b;
    --error-glow: #ef4444;
}

.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f1629 100%) !important;
    color: #f8fafc !important;
    max-width: 100% !important;
    font-family: 'Inter', 'SF Pro Display', system-ui, sans-serif !important;
    position: relative;
}

/* Glassmorphism background texture */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.02"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') !important;
    pointer-events: none;
    z-index: -1;
}

/* Hero Section with Enhanced Glassmorphism */
.hero-section {
    background: var(--glass-bg);
    backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid var(--glass-border);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white !important;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: var(--primary-gradient);
    opacity: 0.1;
    animation: float 6s ease-in-out infinite;
    border-radius: 50%;
}

@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
}

.hero-title {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.5rem !important;
    background: linear-gradient(45deg, #ffffff, var(--neon-accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    z-index: 1;
}

.hero-subtitle {
    font-size: 1.1rem !important;
    opacity: 0.9;
    font-weight: 500;
    position: relative;
    z-index: 1;
}

/* Enhanced Panel Design with Neumorphism */
.analysis-panel {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) saturate(150%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin: 0.75rem 0 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.analysis-panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

/* Modern Chat Interface */
.assistant-chat {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
}

/* 2025 Button Design with Micro-interactions */
.gradio-button {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 8px !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-weight: 600 !important;
    position: relative !important;
    overflow: hidden !important;
}

.gradio-button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    border-color: var(--neon-accent) !important;
}

.gradio-button:active {
    transform: translateY(0px) !important;
}

/* Status Indicators with Glow Effects */
.status-success {
    color: var(--success-glow) !important;
    text-shadow: 0 0 10px var(--success-glow) !important;
}

.status-warning {
    color: var(--warning-glow) !important;
    text-shadow: 0 0 10px var(--warning-glow) !important;
}

.status-error {
    color: var(--error-glow) !important;
    text-shadow: 0 0 10px var(--error-glow) !important;
}

/* Enhanced Typography */
.gradio-html h1, .gradio-html h2, .gradio-html h3 {
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
}

/* Responsive Design for Mobile-First 2025 */
@media (max-width: 768px) {
    .hero-title {
        font-size: 1.8rem !important;
    }
    
    .analysis-panel {
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .gradio-column {
        padding: 0.25rem !important;
    }
}

/* Accessibility Enhancements */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* Focus indicators for accessibility */
.gradio-button:focus-visible,
.gradio-textbox:focus-visible {
    outline: 2px solid var(--neon-accent) !important;
    outline-offset: 2px !important;
}

/* Streaming Status Animation */
.streaming-indicator {
    display: inline-block;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Progress Bar for Streaming */
.progress-bar {
    width: 100%;
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.progress-fill {
    height: 100%;
    background: var(--primary-gradient);
    border-radius: 2px;
    animation: progress 2s infinite;
}

@keyframes progress {
    0% { width: 0%; }
    50% { width: 100%; }
    100% { width: 0%; }
}

@keyframes pulse-glow {
    0%, 100% {
        opacity: 0.6;
        transform: scaleX(1);
    }
    50% {
        opacity: 1;
        transform: scaleX(1.02);
    }
}

@keyframes rotate-glow {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

/* Enhanced Multi-Agent Analysis Styles */
.analysis-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    background: var(--glass-bg);
    border-radius: 12px;
    border: 1px solid var(--glass-border);
}

.action-buttons-row {
    margin: 1.5rem 0;
    gap: 1rem !important;
    align-items: center;
}

.primary-action-btn {
    background: var(--primary-gradient) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transition: all 0.3s ease !important;
    min-height: 44px !important;
}

.primary-action-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
}

.secondary-action-btn {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    color: #94a3b8 !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
    min-height: 44px !important;
}

.secondary-action-btn:hover {
    background: rgba(148, 163, 184, 0.1) !important;
    border-color: var(--neon-accent) !important;
    color: var(--neon-accent) !important;
}

/* Feature Grid Cards */
.feature-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
    gap: 1rem !important;
    margin: 1.5rem 0 !important;
}

.feature-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(15px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 1.5rem 1rem !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.feature-card:hover {
    transform: translateY(-5px) !important;
    border-color: var(--neon-accent) !important;
    box-shadow: 0 10px 25px rgba(0, 245, 255, 0.2) !important;
}

.feature-card::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(0, 245, 255, 0.1), transparent) !important;
    transition: left 0.5s !important;
}

.feature-card:hover::before {
    left: 100% !important;
}

.feature-icon {
    font-size: 2rem !important;
    margin-bottom: 0.5rem !important;
    filter: drop-shadow(0 0 8px var(--neon-accent)) !important;
}

.feature-card h4 {
    color: white !important;
    margin: 0.5rem 0 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

.feature-card p {
    color: #94a3b8 !important;
    margin: 0 !important;
    font-size: 0.9rem !important;
}

/* Agent Progress Dashboard */
.progress-section {
    margin: 2rem 0 !important;
}

.agent-progress-panel {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(15px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    position: relative !important;
    overflow: hidden !important;
}

.agent-progress-panel::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 2px !important;
    background: linear-gradient(90deg, var(--neon-accent), #00ff88, var(--neon-accent)) !important;
    animation: pulse-glow 2s ease-in-out infinite !important;
}

.dashboard-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 1.5rem !important;
    padding-bottom: 1rem !important;
    border-bottom: 1px solid rgba(0, 188, 212, 0.2) !important;
}

.dashboard-header h3 {
    margin: 0 !important;
    color: var(--neon-accent) !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    text-shadow: 0 0 10px rgba(0, 188, 212, 0.3) !important;
}

.agents-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)) !important;
    gap: 1.25rem !important;
    margin-top: 0 !important;
}

.agent-status-item {
    display: flex !important;
    align-items: center !important;
    padding: 1.25rem !important;
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.agent-status-item::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: -100% !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(90deg, transparent, rgba(0, 188, 212, 0.08), transparent) !important;
    transition: left 0.6s ease !important;
}

.agent-status-item:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    border-color: var(--neon-accent) !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 24px rgba(0, 188, 212, 0.15) !important;
}

.agent-status-item:hover::before {
    left: 100% !important;
}

.agent-icon {
    font-size: 1.5rem !important;
    margin-right: 1rem !important;
    filter: drop-shadow(0 0 5px var(--neon-accent)) !important;
}

.agent-info {
    display: flex !important;
    flex-direction: column !important;
    flex-grow: 1 !important;
}

.agent-info strong {
    color: white !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.25rem !important;
}

.agent-icon-large {
    font-size: 2rem !important;
    margin-right: 1.25rem !important;
    filter: drop-shadow(0 0 12px var(--neon-accent)) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 3.5rem !important;
    height: 3.5rem !important;
    background: radial-gradient(circle, rgba(0, 188, 212, 0.15), rgba(0, 188, 212, 0.05)) !important;
    border-radius: 50% !important;
    border: 2px solid rgba(0, 188, 212, 0.4) !important;
    position: relative !important;
    overflow: hidden !important;
}

.agent-details {
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0.75rem !important;
}

.agent-name {
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    text-shadow: 0 0 8px rgba(255, 255, 255, 0.1) !important;
}

.agent-status {
    font-size: 0.9rem !important;
    padding: 0.375rem 0.75rem !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    display: inline-block !important;
    min-width: 120px !important;
    text-align: center !important;
}

.status-indicator {
    background: rgba(148, 163, 184, 0.15) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
}

.status-warning {
    background: rgba(255, 193, 7, 0.15) !important;
    color: #ffc107 !important;
    border: 1px solid rgba(255, 193, 7, 0.3) !important;
}

.status-success {
    background: rgba(40, 167, 69, 0.15) !important;
    color: #28a745 !important;
    border: 1px solid rgba(40, 167, 69, 0.3) !important;
}

/* Results Section */
.results-section {
    margin-top: 2rem !important;
}

.analysis-output-panel {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(15px) saturate(180%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    min-height: 400px !important;
    max-height: 600px !important;
    overflow-y: auto !important;
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    color: #e2e8f0 !important;
}

.analysis-output-panel::-webkit-scrollbar {
    width: 8px !important;
}

.analysis-output-panel::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 4px !important;
}

.analysis-output-panel::-webkit-scrollbar-thumb {
    background: var(--neon-accent) !important;
    border-radius: 4px !important;
    opacity: 0.6 !important;
}

.analysis-output-panel::-webkit-scrollbar-thumb:hover {
    opacity: 1 !important;
}

/* Accessibility Improvements */
.primary-action-btn:focus-visible,
.secondary-action-btn:focus-visible {
    outline: 2px solid var(--neon-accent) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px rgba(0, 245, 255, 0.3) !important;
}

/* Mobile Responsive Enhancements */
@media (max-width: 768px) {
    .feature-grid {
        grid-template-columns: 1fr 1fr !important;
        gap: 0.75rem !important;
    }

    .feature-card {
        padding: 1rem 0.75rem !important;
    }

    .feature-card h4 {
        font-size: 1rem !important;
    }

    .feature-card p {
        font-size: 0.8rem !important;
    }

    .agent-progress-panel {
        grid-template-columns: 1fr !important;
        padding: 1rem !important;
    }

    .agent-status-item {
        padding: 0.75rem !important;
    }

    .action-buttons-row {
        flex-direction: column !important;
        gap: 0.75rem !important;
    }

    .primary-action-btn,
    .secondary-action-btn {
        width: 100% !important;
        margin: 0 !important;
    }

    .analysis-output-panel {
        min-height: 300px !important;
        max-height: 400px !important;
        font-size: 0.8rem !important;
    }
}

@media (max-width: 480px) {
    .feature-grid {
        grid-template-columns: 1fr !important;
    }

    .analysis-header h2 {
        font-size: 1.5rem !important;
    }

    .analysis-header p {
        font-size: 1rem !important;
    }
}
"""


def load_api_spec(file):
    """Load and parse API specification with enhanced validation and verbose RAG processing feedback"""
    global current_spec

    logger.info(f"🚀 RAG Pipeline: Starting document upload process for {file.name if file else 'No file'}")

    if not file:
        logger.warning("❌ RAG Pipeline: No file provided for upload")
        return None, "Upload an API specification to get started", ""

    try:
        # Enhanced input validation with RAG context
        file_path = file.name
        file_size = os.path.getsize(file_path)

        logger.info(f"📄 RAG Pipeline: File validation started")
        logger.info(f"   • File path: {file_path}")
        logger.info(f"   • File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

        # Check file size (max 10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if file_size > MAX_FILE_SIZE:
            error_msg = f"❌ RAG Pipeline: File too large ({file_size:,} bytes). Max: {MAX_FILE_SIZE:,} bytes (10MB)"
            logger.warning(error_msg)
            return None, error_msg, ""

        # Check file extension
        allowed_extensions = {'.json', '.yaml', '.yml'}
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in allowed_extensions:
            error_msg = f"❌ RAG Pipeline: Invalid file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}"
            logger.warning(error_msg)
            return None, error_msg, ""

        logger.info(f"✅ RAG Pipeline: File validation passed - {file_ext} format detected")

        # Content reading with RAG processing context
        logger.info(f"📖 RAG Pipeline: Reading and parsing document content...")
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"   • Content length: {len(content):,} characters")
        logger.info(f"   • Estimated tokens: ~{len(content.split()):,} words")

        # Format-specific parsing
        if file.name.endswith((".yaml", ".yml")):
            logger.info("🔄 RAG Pipeline: Parsing YAML document structure...")
            spec = yaml.safe_load(content)
        else:
            logger.info("🔄 RAG Pipeline: Parsing JSON document structure...")
            spec = json.loads(content)

        # Document structure analysis for RAG
        spec_keys = list(spec.keys()) if isinstance(spec, dict) else []
        logger.info(f"🔍 RAG Pipeline: Document structure analysis:")
        logger.info(f"   • Root keys found: {spec_keys}")
        logger.info(f"   • Document type: {'Dictionary/Object' if isinstance(spec, dict) else type(spec).__name__}")

        current_spec = spec

        # Handle both OpenAPI 3.0 format and custom format
        if "openapi" in spec or "swagger" in spec:
            # Standard OpenAPI/Swagger format
            info = spec.get("info", {})
            servers = spec.get("servers", [])
            paths = spec.get("paths", {})

            api_title = info.get("title", "API Documentation")
            api_version = info.get("version", "Unknown")
            api_description = info.get("description", "No description provided")
            base_url = (
                servers[0].get("url", "Not specified") if servers else "Not specified"
            )
            endpoint_count = len(paths)
        else:
            # Custom format (like your JSON)
            api_title = spec.get("api_name", spec.get("title", "API Documentation"))
            api_version = spec.get("version", "Unknown")
            api_description = spec.get("description", "No description provided")
            base_url = spec.get("base_url", "Not specified")
            # Handle custom 'endpoints' array
            endpoints_data = spec.get("endpoints", [])
            endpoint_count = len(endpoints_data)

            # Convert custom format to OpenAPI-like structure for processing
            paths = {}
            for endpoint in endpoints_data:
                path = endpoint.get("path", "/")
                method = endpoint.get("method", "get").lower()
                if path not in paths:
                    paths[path] = {}
                paths[path][method] = {
                    "summary": endpoint.get("description", "No description"),
                    "description": endpoint.get("description", "No description"),
                }

        logger.info(
            f"Building API info display - Title: {api_title}, Version: {api_version}, Endpoints: {endpoint_count}"
        )

        api_info = f"""## 📋 **{api_title}**

**Version**: {api_version}  
**Base URL**: {base_url}  
**Description**: {api_description}

<h3>🎯 **Available Endpoints** ({endpoint_count} total)</h3>
"""

        # Add endpoint details
        methods_found = 0
        for path, methods in paths.items():
            api_info += f"\n**`{path}`**\n"
            if isinstance(methods, dict):
                for method, details in methods.items():
                    methods_found += 1
                    if isinstance(details, dict):
                        summary = details.get(
                            "summary", details.get("description", "No summary")
                        )
                        api_info += f"- **{method.upper()}** - {summary}\n"

        # Enhanced RAG-specific logging for API structure analysis
        logger.info(f"🔍 RAG Pipeline: API Structure Analysis Complete")
        logger.info(f"   • Total endpoints: {endpoint_count}")
        logger.info(f"   • HTTP methods found: {methods_found}")
        logger.info(f"   • API title: {api_title}")
        logger.info(f"   • API version: {api_version}")

        # RAG content preparation metrics
        estimated_chunks = max(1, len(content) // 1000)  # Rough chunk estimation
        logger.info(f"📊 RAG Pipeline: Content Preparation Metrics")
        logger.info(f"   • Estimated text chunks: ~{estimated_chunks}")
        logger.info(f"   • Document complexity: {'High' if endpoint_count > 10 else 'Medium' if endpoint_count > 5 else 'Low'}")
        logger.info(f"   • Content density: {len(content) / max(1, endpoint_count):.0f} chars/endpoint")

        # Add additional info for both formats
        if "openapi" in spec or "swagger" in spec:
            # OpenAPI format contact info
            spec_info = spec.get("info", {})
            if "contact" in spec_info:
                contact = spec_info["contact"]
                api_info += f"\n<h3>📧 **Contact Information**</h3>\n"
                if contact.get("name"):
                    api_info += f"**Name**: {contact['name']}\n"
                if contact.get("email"):
                    api_info += f"**Email**: {contact['email']}\n"
                if contact.get("url"):
                    api_info += f"**URL**: {contact['url']}\n"
        else:
            # Custom format additional info
            if spec.get("authentication"):
                auth_info = spec["authentication"]
                api_info += f"\n<h3>🔐 **Authentication**</h3>\n"
                api_info += f"**Type**: {auth_info.get('type', 'Not specified')}\n"
                if auth_info.get("description"):
                    api_info += f"**Details**: {auth_info['description']}\n"

            if spec.get("rate_limiting"):
                api_info += f"\n<h3>⚡ **Rate Limiting**</h3>\n"
                api_info += f"{spec['rate_limiting']}\n"

        # Final RAG pipeline completion
        spec_json = json.dumps(spec, indent=2)

        # RAG Pipeline completion metrics
        logger.info(f"🎯 RAG Pipeline: Document Processing Complete")
        logger.info(f"   • Final JSON size: {len(spec_json):,} characters")
        logger.info(f"   • Ready for vector indexing: Yes")
        logger.info(f"   • Document status: Successfully parsed and validated")
        logger.info(f"   • Next steps: Document ready for chunking and embedding")

        # Enhanced status with RAG context
        status = f"✅ RAG Pipeline: Document loaded and ready for AI assistant ({endpoint_count} endpoints indexed)"

        logger.info(f"🚀 RAG Pipeline: Upload process completed successfully")
        return spec_json, api_info, status

    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ RAG Pipeline: Document processing failed - {error_message}")
        logger.error(f"   • Error type: {type(e).__name__}")
        logger.error(f"   • Processing stage: Document parsing and validation")
        logger.error(f"   • Impact: RAG assistant will not be available for this document")

        return (
            None,
            f"❌ **RAG Pipeline Error**: Failed to process document for AI assistant\n\n**Details**: {error_message}\n\n**Solution**: Please ensure your document is a valid JSON or YAML API specification.",
            "❌ RAG Pipeline: Document processing failed",
        )


def load_demo():
    """Load demo Pet Store API with logging"""
    global current_spec

    logger.info("Loading Pet Store demo API specification")

    demo_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Pet Store API Demo",
            "version": "1.0.0",
            "description": "A sample Pet Store API for demonstration",
        },
        "servers": [{"url": "https://petstore.swagger.io/v2"}],
        "paths": {
            "/pets": {
                "get": {"summary": "List all pets"},
                "post": {"summary": "Create a new pet"},
            },
            "/pets/{petId}": {
                "get": {"summary": "Get pet by ID"},
                "delete": {"summary": "Delete a pet"},
            },
        },
    }

    current_spec = demo_spec

    # Log demo spec details
    demo_endpoints = len(demo_spec.get("paths", {}))
    demo_methods = sum(len(methods) for methods in demo_spec.get("paths", {}).values())
    logger.info(
        f"Demo API loaded - {demo_endpoints} endpoints with {demo_methods} HTTP methods total"
    )

    # Generate display info
    api_info = """## 📋 **Pet Store API Demo**

**Version**: 1.0.0  
**Base URL**: https://petstore.swagger.io/v2  
**Description**: A sample Pet Store API for demonstration

<h3>🎯 **Available Endpoints** (2 total)</h3>

**`/pets`**
- **GET** - List all pets
- **POST** - Create a new pet

**`/pets/{petId}`**
- **GET** - Get pet by ID  
- **DELETE** - Delete a pet
"""

    spec_json = json.dumps(demo_spec, indent=2)
    status = "✅ Demo API loaded successfully"

    logger.info("Pet Store demo API successfully loaded and formatted for display")

    return spec_json, api_info, status




def enhance_analysis_formatting(content):
    """Rich, professional formatting with enhanced content density"""
    if not content or len(content.strip()) < 10:
        return content

    enhanced_content = content

    # 1. Format Overall Score with rich visual elements
    overall_score_pattern = r"\*\*Overall Score:\*\* (\d+)/100"

    def format_overall_score(match):
        score = int(match.group(1))
        status_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        status_text = (
            "Excellent"
            if score >= 80
            else "Good" if score >= 60 else "Needs Improvement"
        )

        # Create visual progress bar
        progress_blocks = "█" * (score // 10) + "░" * (10 - score // 10)

        # Add quality indicators
        indicators = []
        if score >= 90:
            indicators = [
                "🏆 Premium Quality",
                "✨ Production Ready",
                "🎯 Best Practices",
            ]
        elif score >= 80:
            indicators = [
                "✅ High Quality",
                "🚀 Well Designed",
                "📝 Good Documentation",
            ]
        elif score >= 60:
            indicators = ["⚠️ Adequate", "🔧 Needs Refinement", "📋 Some Issues"]
        else:
            indicators = ["❌ Poor Quality", "🚨 Major Issues", "⚡ Needs Work"]

        return f"""# 📊 API Quality Analysis Report

## {status_emoji} Overall Score: {score}/100 - {status_text}

**Visual Progress:** `{progress_blocks}` ({score}%)

<h3>Quality Indicators:</h3>
{chr(10).join(f"• {indicator}" for indicator in indicators)}

<h3>Analysis Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</h3>

---
"""

    enhanced_content = re.sub(
        overall_score_pattern, format_overall_score, enhanced_content
    )

    # 2. Format detailed score breakdowns with rich details
    score_patterns = {
        r"- \*\*(\w+):\*\* (\d+)/100 - ([^\n]+)": lambda m: format_rich_score_item(
            m.group(1), int(m.group(2)), m.group(3)
        ),
        r"• \*\*(\w+):\*\* (\d+)/100 - ([^\n]+)": lambda m: format_rich_score_item(
            m.group(1), int(m.group(2)), m.group(3)
        ),
        r"◦ \*\*(\w+):\*\* (\d+)/100 - ([^\n]+)": lambda m: format_rich_score_item(
            m.group(1), int(m.group(2)), m.group(3)
        ),
    }

    def format_rich_score_item(category, score, description):
        icons = {
            "Completeness": "🔧",
            "Documentation": "📚",
            "Security": "🔒",
            "Usability": "👥",
            "Standards": "📏",
            "Performance": "⚡",
            "Compliance": "📏",
        }

        impact_levels = {
            "Completeness": "Foundation",
            "Documentation": "Developer Experience",
            "Security": "Trust & Safety",
            "Usability": "Adoption",
            "Standards": "Maintainability",
            "Performance": "Scalability",
            "Compliance": "Reliability",
        }

        icon = icons.get(category, "📊")
        impact = impact_levels.get(category, "Quality")

        if score >= 80:
            status = "✅ Excellent"
            grade = "A"
            bar = "████████▓░"
        elif score >= 60:
            status = "⚠️ Good"
            grade = "B"
            bar = "██████▓▓░░"
        else:
            status = "❌ Needs Work"
            grade = "C"
            bar = "████▓▓▓▓░░"

        return f"""
<h3>{icon} {category}: {score}/100 {status} (Grade: {grade})</h3>

**Impact Area:** {impact}  
**Progress:** `{bar}` {score}%  
**Analysis:** {description}

"""

    for pattern, formatter in score_patterns.items():
        enhanced_content = re.sub(pattern, formatter, enhanced_content)

    # 3. Format critical issues with enhanced structure
    def format_critical_issues(content):
        # Format numbered critical issues
        issue_pattern = r"^(\d+\.\s+\*\*Issue:\*\*\s+.+)$"

        def format_numbered_issue(match):
            issue_text = match.group(1)
            issue_number = re.match(r"^(\d+)\.", issue_text).group(1)
            clean_text = re.sub(r"\d+\.\s+\*\*Issue:\*\*\s+", "", issue_text)

            priority_labels = ["🔥 Critical", "⚠️ High", "📋 Medium", "💡 Low"]
            priority = (
                priority_labels[min(int(issue_number) - 1, 3)]
                if issue_number.isdigit()
                else "📋 Issue"
            )

            return f"""
#### {priority} - Issue #{issue_number}: {clean_text}

**Severity:** Priority {issue_number}  
**Category:** API Design Flaw
"""

        content = re.sub(
            issue_pattern, format_numbered_issue, content, flags=re.MULTILINE
        )

        # Format sub-bullet details with enhanced styling
        sub_bullet_pattern = r"^\s*-\s+\*\*([^*]+):\*\*\s+(.+)$"

        def format_issue_details(match):
            label = match.group(1)
            detail = match.group(2)

            label_icons = {
                "Location": "📍",
                "Impact": "💥",
                "Fix": "🔧",
                "Priority": "⭐",
                "Reason": "💭",
                "Implementation": "🛠️",
            }

            icon = label_icons.get(label, "•")
            return f"**{icon} {label}:** {detail}"

        content = re.sub(
            sub_bullet_pattern, format_issue_details, content, flags=re.MULTILINE
        )
        return content

    enhanced_content = format_critical_issues(enhanced_content)

    # 4. Add summary statistics after executive summary
    def add_analysis_stats(content):
        # Count various elements for statistics
        issue_count = len(re.findall(r"#### .* - Issue #\d+", content))
        recommendation_count = len(re.findall(r"\*\*Fix:\*\*", content))

        stats_section = f"""
## 📈 Analysis Summary

| Metric | Count | Status |
|--------|--------|--------|
| Critical Issues | {issue_count} | {"🔴 Action Required" if issue_count > 2 else "🟡 Review Needed" if issue_count > 0 else "🟢 Good"} |
| Recommendations | {recommendation_count} | {"📝 Improvement Plan Available" if recommendation_count > 0 else "✅ No Issues Found"} |
| Analysis Depth | Comprehensive | 🔍 Full Coverage |

---
"""

        # Insert after executive summary if it exists
        if "## 🎯 Executive Summary" in content:
            content = content.replace(
                "## 🎯 Executive Summary", stats_section + "## 🎯 Executive Summary"
            )
        else:
            # Insert near the beginning
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("## ") and "🎯" not in line:
                    lines.insert(i, stats_section)
                    break
            content = "\n".join(lines)

        return content

    enhanced_content = add_analysis_stats(enhanced_content)

    # 5. Format main section headers with better styling
    header_replacements = {
        r"<h3>🎯 Executive Summary</h3>": "## 🎯 Executive Summary\n*Key findings and overall assessment*",
        r"<h3>🚨 Critical Issues \(Priority Order\)</h3>": "## 🚨 Critical Issues (Priority Order)\n*Issues requiring immediate attention*",
        r"<h3>🔍 Detailed Analysis</h3>": "## 🔍 Detailed Analysis\n*Comprehensive evaluation results*",
        r"<h3>(.+)</h3>": r"## \1",
    }

    for pattern, replacement in header_replacements.items():
        enhanced_content = re.sub(pattern, replacement, enhanced_content)

    # 6. Format code blocks with language hints
    enhanced_content = re.sub(
        r"```(yaml|json|javascript|python)?\n(.*?)\n```",
        lambda m: f"```{m.group(1) or 'yaml'}\n{m.group(2)}\n```\n*Example {m.group(1) or 'configuration'} implementation*",
        enhanced_content,
        flags=re.DOTALL,
    )

    # 7. Add visual separators and improve spacing
    enhanced_content = re.sub(r"\n---\n", "\n\n---\n\n", enhanced_content)
    enhanced_content = re.sub(r"\n{4,}", "\n\n", enhanced_content)
    enhanced_content = re.sub(r"\*\*([^*]+):\*\*", r"**\1:**", enhanced_content)

    # 8. Add final footer with action items
    footer = """
---

## 🎯 Next Steps

1. **Address Critical Issues:** Start with Priority 1 items for immediate impact
2. **Implement Recommendations:** Follow the provided fixes and improvements  
3. **Monitor Progress:** Re-run analysis after implementing changes
4. **Documentation:** Update API documentation based on findings

*This analysis was generated using AI-powered evaluation tools. For detailed implementation guidance, consult the specific recommendations above.*
"""

    enhanced_content += footer

    return enhanced_content.strip()




async def start_analysis():
    """Start API analysis (non-streaming fallback)"""
    if not current_spec:
        return "❌ Please upload an API specification first", "", None

    try:
        logger.info("Starting API analysis via backend")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/analyze",
                json={
                    "openapi_spec": current_spec,
                    "focus_areas": [
                        "security",
                        "performance",
                        "documentation",
                        "completeness",
                        "standards",
                    ],
                },
                timeout=DEFAULT_TIMEOUT,
            )

            if response.status_code == 200:
                result = response.json()
                analysis = result.get("analysis", "Analysis completed successfully")
                # Apply enhanced formatting to the analysis
                enhanced_analysis = enhance_analysis_formatting(analysis)
                logger.info("Analysis completed successfully")
                return enhanced_analysis, "Analysis completed", None
            else:
                error_msg = handle_api_error(response.status_code, "API analysis")
                return error_msg, "", None

    except httpx.TimeoutException:
        return log_error_with_context(
            httpx.TimeoutException("Request timed out"),
            "API analysis timeout",
            f"❌ Analysis timed out after {DEFAULT_TIMEOUT} seconds. Please try again."
        ), "", None
    except httpx.RequestError as e:
        return log_error_with_context(e, "API analysis connection", "❌ Cannot connect to backend service. Please check if it's running."), "", None
    except Exception as e:
        return log_error_with_context(e, "API analysis"), "", None


def run_rag_pipeline_demo():
    """Run the complete RAG v2 pipeline demo for educational purposes"""
    import subprocess
    import os

    try:
        # Change to the project directory to run the test
        original_cwd = os.getcwd()

        # Run the pipeline test and capture output
        result = subprocess.run(
            ["poetry", "run", "python", "test_full_rag_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Format the output for display
        output_lines = []
        output_lines.append("=" * 60)
        output_lines.append("🚀 RAG v2 Pipeline Demo Execution")
        output_lines.append("=" * 60)
        output_lines.append("")

        if result.returncode == 0:
            output_lines.append("✅ Pipeline executed successfully!")
            output_lines.append("")
            output_lines.append("📊 EXECUTION OUTPUT:")
            output_lines.append("-" * 40)
            output_lines.extend(result.stdout.split("\n"))

            if result.stderr:
                output_lines.append("")
                output_lines.append("⚠️ WARNINGS/INFO:")
                output_lines.append("-" * 40)
                output_lines.extend(result.stderr.split("\n"))
        else:
            output_lines.append("❌ Pipeline execution encountered issues")
            output_lines.append("")
            output_lines.append("📊 STDOUT:")
            output_lines.append("-" * 40)
            if result.stdout:
                output_lines.extend(result.stdout.split("\n"))
            else:
                output_lines.append("(No stdout output)")

            output_lines.append("")
            output_lines.append("❌ STDERR:")
            output_lines.append("-" * 40)
            if result.stderr:
                output_lines.extend(result.stderr.split("\n"))
            else:
                output_lines.append("(No stderr output)")

        output_lines.append("")
        output_lines.append("=" * 60)
        output_lines.append("🎯 Demo Complete - Review the output above to understand:")
        output_lines.append("   • How each RAG v2 component initializes")
        output_lines.append("   • Component interaction and data flow")
        output_lines.append("   • Performance metrics and evaluation scores")
        output_lines.append("   • Error handling and fallback mechanisms")
        output_lines.append("=" * 60)

        return "\n".join(output_lines)

    except subprocess.TimeoutExpired:
        return """⏰ PIPELINE DEMO TIMEOUT

The RAG v2 pipeline demo took longer than expected to complete.
This might be due to:
• First-time model download (embeddings)
• System resource constraints
• Network connectivity issues

Try running the demo again, or check the backend logs for more details."""

    except Exception as e:
        return f"""❌ PIPELINE DEMO ERROR

An error occurred while running the RAG v2 pipeline demo:
{str(e)}

This is an educational demo. The error helps understand:
• Real-world deployment challenges
• Error handling mechanisms
• System dependencies

Check your environment setup and try again."""


def clear_pipeline_output():
    """Clear the pipeline output display"""
    return "Click 'Run RAG v2 Pipeline Demo' to see the complete pipeline in action..."


# Multi-Agent Streaming Functions
def stream_multiagent_analysis_with_dashboard(openapi_spec_content):
    """Stream multi-agent analysis with simulated real-time dashboard updates for Gradio"""

    if not openapi_spec_content:
        yield "❌ Error: Please upload an OpenAPI specification first!", generate_dashboard_html()
        return

    # Handle both string and dict formats
    if isinstance(openapi_spec_content, str):
        if not openapi_spec_content.strip():
            yield "❌ Error: Please upload an OpenAPI specification first!", generate_dashboard_html()
            return
        try:
            spec_dict = json.loads(openapi_spec_content)
        except json.JSONDecodeError:
            yield "❌ Error: Invalid JSON format in OpenAPI specification!", generate_dashboard_html()
            return
    elif isinstance(openapi_spec_content, dict):
        spec_dict = openapi_spec_content
    else:
        yield "❌ Error: Invalid OpenAPI specification format!", generate_dashboard_html()
        return

    BASE_URL = "http://localhost:8080"

    try:
        # Initialize agent statuses
        agent_statuses = {
            'SecurityAnalyst': '⏳ Waiting',
            'DocumentationReviewer': '⏳ Waiting',
            'PerformanceEngineer': '⏳ Waiting',
            'IntegrationSpecialist': '⏳ Waiting'
        }

        # Initialize output
        full_output = ["🚀 **Starting Multi-Agent Analysis...**\n"]
        full_output.append("📋 **Initializing Agents:** SecurityAnalyst, DocumentationReviewer, PerformanceEngineer, IntegrationSpecialist\n\n")

        # Phase 1: Initial state
        yield ''.join(full_output), generate_dashboard_html(agent_statuses)

        # Phase 2: Show agents starting (simulated progressive updates)
        import time
        agents = ['SecurityAnalyst', 'DocumentationReviewer', 'PerformanceEngineer', 'IntegrationSpecialist']

        for i, agent in enumerate(agents):
            agent_statuses[agent] = '🔄 Running'
            full_output.append(f"🤖 **{agent}**: Starting analysis...\n")
            yield ''.join(full_output), generate_dashboard_html(agent_statuses)
            time.sleep(0.5)  # Small delay for visual effect

        # Phase 3: Run actual analysis (non-streaming for now, but with dashboard updates)
        full_output.append("\n🔄 **Running comprehensive analysis...**\n\n")
        yield ''.join(full_output), generate_dashboard_html(agent_statuses)

        # Make the actual API call
        import requests
        analysis_request = {
            "openapi_spec": spec_dict,
            "analysis_options": {
                "include_security_analysis": True,
                "include_performance_analysis": True,
                "include_documentation_analysis": True,
                "detailed_output": True
            }
        }

        try:
            response = requests.post(
                f"{BASE_URL}/analyze-agentic",
                json=analysis_request,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()

                # Phase 4: Show agents completing
                for agent in agents:
                    agent_statuses[agent] = '✅ Complete'
                    full_output.append(f"✅ **{agent}**: Analysis completed\n")
                    yield ''.join(full_output), generate_dashboard_html(agent_statuses)
                    time.sleep(0.3)

                # Phase 5: Show final results
                full_output.append(f"\n🎉 **Multi-Agent Analysis Complete!**\n\n")

                # Add the actual analysis results
                analysis_content = result.get("consolidated_analysis", "No analysis content available")
                full_output.append("---\n\n## 📋 Consolidated Analysis Report\n\n")
                full_output.append(analysis_content)

                yield ''.join(full_output), generate_dashboard_html(agent_statuses)

            else:
                error_msg = f"❌ Error: Backend returned status {response.status_code}"
                yield error_msg, generate_dashboard_html()

        except requests.exceptions.ConnectionError:
            error_msg = """❌ **Connection Error**: Cannot connect to backend server.

🔧 **Quick Fix:**
1. Make sure the backend is running: `poetry run python -m api.main`
2. Check if port 8080 is available
3. Verify the API server started successfully

⚡ The backend should be accessible at http://localhost:8080"""
            yield error_msg, generate_dashboard_html()

        except requests.exceptions.Timeout:
            error_msg = """⏰ **Analysis Timeout**: The multi-agent analysis took too long to complete."""
            yield error_msg, generate_dashboard_html()

    except Exception as e:
        logger.error(f"Streaming multi-agent analysis error: {str(e)}")
        error_msg = f"""❌ **Unexpected Error**: {str(e)}

🔍 **Troubleshooting:**
1. Check the backend logs for details
2. Ensure the OpenAPI spec is valid JSON/YAML
3. Try refreshing and running again"""
        yield error_msg, generate_dashboard_html()


def generate_dashboard_html(agent_statuses=None):
    """Generate dynamic dashboard HTML with current agent statuses"""
    if agent_statuses is None:
        agent_statuses = {
            'SecurityAnalyst': '⏳ Waiting',
            'DocumentationReviewer': '⏳ Waiting',
            'PerformanceEngineer': '⏳ Waiting',
            'IntegrationSpecialist': '⏳ Waiting'
        }

    agent_icons = {
        'SecurityAnalyst': '🔒',
        'DocumentationReviewer': '📝',
        'PerformanceEngineer': '⚡',
        'IntegrationSpecialist': '🔗'
    }

    dashboard_items = []
    for agent, status in agent_statuses.items():
        icon = agent_icons.get(agent, '🤖')
        status_class = 'status-success' if '✅' in status else 'status-warning' if '🔄' in status else 'status-indicator'

        dashboard_items.append(f"""
            <div class="agent-status-item">
                <div class="agent-icon-large">{icon}</div>
                <div class="agent-details">
                    <div class="agent-name">{agent}</div>
                    <div class="agent-status {status_class}">{status}</div>
                </div>
            </div>
        """)

    return f"""
        <div class="agent-progress-panel">
            <div class="dashboard-header">
                <h3 style="margin: 0 0 15px 0; color: #00bcd4; font-size: 16px; font-weight: 600;">
                    🤖 Agent Progress Dashboard
                </h3>
            </div>
            <div class="agents-grid">
                {''.join(dashboard_items)}
            </div>
        </div>
    """


# Legacy function for backward compatibility
async def stream_multiagent_analysis(openapi_spec_content):
    """Legacy streaming function - now replaced by generator version"""
    # This is kept for any existing references but should not be used
    return "❌ This function has been replaced with real-time dashboard updates"


def multiagent_analyze_api_with_dashboard(openapi_spec_content):
    """Gradio wrapper for streaming multi-agent analysis with real-time dashboard updates"""
    try:
        # Use the new generator-based streaming function for real-time updates
        for output, dashboard in stream_multiagent_analysis_with_dashboard(openapi_spec_content):
            yield output, dashboard
    except Exception as e:
        logger.error(f"Multi-agent analysis wrapper error: {str(e)}")
        yield f"❌ Error running multi-agent analysis: {str(e)}", generate_dashboard_html()


# Legacy wrapper for backward compatibility
def multiagent_analyze_api(openapi_spec_content):
    """Legacy wrapper - now uses dashboard version"""
    # Convert generator to single result for compatibility
    try:
        final_output = None
        for output, dashboard in stream_multiagent_analysis_with_dashboard(openapi_spec_content):
            final_output = output
        return final_output or "❌ No analysis result generated"
    except Exception as e:
        logger.error(f"Multi-agent analysis wrapper error: {str(e)}")
        return f"❌ Error running multi-agent analysis: {str(e)}"


def clear_multiagent_output():
    """Clear multi-agent analysis output (legacy)"""
    return ""


def chat_with_api(message, history, model_name="gpt-4o-mini"):
    """Enhanced chat function with proper input validation"""

    # Enhanced input validation
    if not message or not message.strip():
        logger.warning("Empty message received in chat")
        return history, ""

    message = message.strip()

    # Check message length
    MAX_MESSAGE_LENGTH = 4000  # Reasonable limit for API processing
    if len(message) > MAX_MESSAGE_LENGTH:
        logger.warning(f"Message too long: {len(message)} chars, max: {MAX_MESSAGE_LENGTH}")
        history.append([
            message[:100] + "...",
            f"❌ Message too long ({len(message)} characters). Please keep messages under {MAX_MESSAGE_LENGTH} characters."
        ])
        return history, ""

    # Check for potential security issues (basic validation)
    suspicious_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
    if any(pattern in message.lower() for pattern in suspicious_patterns):
        logger.warning(f"Suspicious content detected in message: {message[:100]}")
        history.append([
            message,
            "❌ Message contains potentially unsafe content. Please rephrase your question."
        ])
        return history, ""

    # Enhanced RAG pipeline initialization logging
    logger.info(f"🔗 RAG Pipeline: Chat request initiated")
    logger.info(f"   • Message length: {len(message)} characters")
    logger.info(f"   • Message preview: {message[:100]}{'...' if len(message) > 100 else ''}")

    if not current_spec:
        logger.warning("❌ RAG Pipeline: No API specification loaded for context")
        history.append([
            message,
            "Please upload an API specification first to enable AI assistance.",
        ])
        return history, ""

    # Enhanced spec context analysis for RAG
    spec_title = current_spec.get("info", {}).get("title", "Unknown API")
    spec_endpoints = len(current_spec.get("paths", {}))
    spec_components = len(current_spec.get("components", {}).get("schemas", {}))

    logger.info(f"🔍 RAG Pipeline: Context Analysis")
    logger.info(f"   • API title: {spec_title}")
    logger.info(f"   • Available endpoints: {spec_endpoints}")
    logger.info(f"   • Schema components: {spec_components}")
    logger.info(f"   • Spec format: {'OpenAPI 3.0' if 'openapi' in current_spec else 'Custom/Swagger'}")

    try:
        # Enhanced RAG backend initialization
        logger.info(f"🚀 RAG Pipeline: Initializing backend request")
        logger.info(f"   • Backend URL: {API_BASE_URL}/rag-query")
        logger.info(f"   • Timeout configuration: {RAG_TIMEOUT}s")
        logger.info(f"   • Request payload size: ~{len(str(current_spec)):,} characters")
        logger.debug(f"current_spec type: {type(current_spec)}, not None: {current_spec is not None}")
        if current_spec:
            logger.debug(f"current_spec keys: {list(current_spec.keys()) if isinstance(current_spec, dict) else 'Not a dict'}")

        import requests

        response = requests.post(
            f"{API_BASE_URL}/rag-query",
            json={
                "question": message,
                "openapi_spec": current_spec,
                "model_name": model_name,  # Selected model from dropdown
                "max_tokens": 1000  # Limit token generation for concise responses
            },
            timeout=RAG_TIMEOUT,  # Configurable timeout for RAG processing
        )

        # Enhanced RAG response processing
        logger.info(f"📡 RAG Pipeline: Backend response received")
        logger.info(f"   • HTTP Status: {response.status_code}")
        logger.info(f"   • Response size: {len(response.content):,} bytes")

        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "Sorry, I couldn't generate a response.")
            metadata = result.get("metadata", {})

            # Enhanced RAG pipeline metrics logging
            logger.info(f"🎯 RAG Pipeline: Processing Metrics")
            logger.info(f"   • Answer length: {len(answer)} characters")
            logger.info(f"   • Retrieved contexts: {metadata.get('contexts_retrieved', 'Unknown')}")
            logger.info(f"   • Processing time: {metadata.get('processing_time', 'Unknown')}s")
            logger.info(f"   • Vector search results: {metadata.get('vector_results', 'Unknown')}")
            logger.info(f"   • Embedding model: {metadata.get('embedding_model', 'Unknown')}")
            logger.info(f"   • Chunking strategy: {metadata.get('chunking_strategy', 'Unknown')}")

            # Enhanced RAG features display
            enhanced_features = []

            # Enhanced RAG Pipeline Details for User Display

            # Vector Store and Embedding Information
            embedding_model = metadata.get("embedding_model", "BAAI/bge-m3")
            vector_store_type = metadata.get("vector_store", "ChromaDB")
            enhanced_features.append(f"🧠 **Embedding Model**: {embedding_model}")
            enhanced_features.append(f"🗃️ **Vector Store**: {vector_store_type}")

            # Document Processing Details
            chunking_strategy = metadata.get("chunking_strategy", "Recursive Character Text Splitter")
            chunk_size = metadata.get("chunk_size", 1000)
            chunk_overlap = metadata.get("chunk_overlap", 200)
            enhanced_features.append(f"✂️ **Chunking Strategy**: {chunking_strategy}")
            enhanced_features.append(f"📏 **Chunk Size**: {chunk_size} chars (overlap: {chunk_overlap})")

            # Cache status
            cache_hit = result.get("cache_hit", False)
            if cache_hit:
                enhanced_features.append("💾 **Cache Hit** - Response retrieved from semantic cache")
            else:
                enhanced_features.append("🔍 **Fresh Search** - New vector similarity search performed")

            # Retrieved contexts with more detail
            retrieved_contexts = result.get("retrieved_contexts", [])
            context_sources = result.get("context_sources", [])
            total_context_chars = sum(len(str(ctx)) for ctx in retrieved_contexts) if retrieved_contexts else 0

            if retrieved_contexts:
                enhanced_features.append(f"📝 **Retrieved Chunks**: {len(retrieved_contexts)} chunks ({total_context_chars:,} chars)")
                if context_sources:
                    enhanced_features.append(f"📂 **Content Sources**: {', '.join(context_sources[:3])}{'...' if len(context_sources) > 3 else ''}")

                # Similarity scores if available
                similarity_scores = metadata.get("similarity_scores", [])
                if similarity_scores:
                    avg_similarity = sum(similarity_scores) / len(similarity_scores)
                    enhanced_features.append(f"🎯 **Avg Similarity**: {avg_similarity:.3f} (best: {max(similarity_scores):.3f})")

            # Evaluation metrics
            evaluation_metrics = result.get("evaluation_metrics", {})
            if evaluation_metrics and evaluation_metrics != {}:
                overall_score = evaluation_metrics.get("overall_score")
                if overall_score:
                    enhanced_features.append(f"⭐ **Quality Score**: {overall_score:.2f}")

                # Individual metrics
                relevancy = evaluation_metrics.get("answer_relevancy_score")
                faithfulness = evaluation_metrics.get("faithfulness_score")
                contextual = evaluation_metrics.get("contextual_relevancy_score")

                if relevancy:
                    enhanced_features.append(f"📊 **Relevancy**: {relevancy:.2f}")
                if faithfulness:
                    enhanced_features.append(f"🎯 **Faithfulness**: {faithfulness:.2f}")
                if contextual:
                    enhanced_features.append(f"🔗 **Context Relevancy**: {contextual:.2f}")

            # Performance metrics
            response_time = result.get("response_time", 0)
            model_used = result.get("model_used", "unknown")
            enhanced_features.append(f"⏱️ **Response Time**: {response_time:.2f}s")
            enhanced_features.append(f"🤖 **Model**: {model_used}")

            # Token usage
            if metadata:
                total_tokens = metadata.get("total_tokens")
                if total_tokens:
                    enhanced_features.append(f"🔢 **Tokens Used**: {total_tokens}")

            # Add vector store creation status
            vector_db_status = metadata.get("vector_db_status", "Ready")
            index_size = metadata.get("index_size", "Unknown")
            enhanced_features.append(f"🔢 **Vector Index**: {vector_db_status} ({index_size} vectors)")

            # Format enhanced answer with comprehensive RAG features
            features_display = ""
            if enhanced_features:
                features_display = f"\n\n---\n**🚀 RAG Pipeline Details:**\n" + "\n".join([f"• {feature}" for feature in enhanced_features])

            enhanced_answer = f"🤖 {answer}{features_display}"

            # Log response metrics with enhanced details
            logger.info(
                f"RAG response received - Answer length: {len(answer)} chars, "
                f"Cache hit: {cache_hit}, Retrieved contexts: {len(retrieved_contexts)}, "
                f"Has evaluation: {bool(evaluation_metrics)}"
            )

            try:
                history.append([message, enhanced_answer])
                logger.info(f"Successfully appended to history - New history length: {len(history)} entries")
            except Exception as append_error:
                logger.error(f"Failed to append to history: {append_error}")
                raise
        else:
            error_msg = handle_api_error(response.status_code, "RAG query")
            history.append([message, error_msg])

    except requests.exceptions.Timeout:
        error_msg = log_error_with_context(
            requests.exceptions.Timeout("Request timed out"),
            "RAG query timeout",
            f"❌ Chat request timed out after {RAG_TIMEOUT} seconds. Please try again."
        )
        history.append([message, error_msg])
    except requests.exceptions.ConnectionError:
        error_msg = log_error_with_context(
            requests.exceptions.ConnectionError("Connection failed"),
            "RAG query connection",
            "❌ Cannot connect to backend service. Please check if it's running."
        )
        history.append([message, error_msg])
    except Exception as e:
        error_msg = log_error_with_context(e, "RAG query")
        history.append([message, error_msg])

    return history, ""


def create_complete_interface():
    """Create the complete interface with both evaluation and assistant tabs"""
    with gr.Blocks(
        css=CUSTOM_CSS, title="APISage - AI-Powered API Analysis & Assistant"
    ) as app:

        # Hero Section
        gr.HTML(
            """
        <div class="hero-section">
            <div class="hero-title">🚀 APISage Complete</div>
            <div class="hero-subtitle">Stage 1: API Analysis & Evaluation • Stage 2: RAG Assistant</div>
        </div>
        """
        )

        # API Specification section only
        with gr.Row():
            gr.HTML("<h3>📁 API Specification</h3>")
            with gr.Row():
                file_input = gr.File(
                    label="Upload JSON/YAML",
                    file_types=[".json", ".yaml", ".yml"],
                    scale=3,
                )
                demo_btn = gr.Button("🐾 Demo", variant="secondary", scale=1)

        with gr.Row():
            status_display = gr.Textbox(
                label="Upload Status",
                value="Upload an API specification to get started",
                interactive=False,
                scale=1,
            )

        # Main tabs
        with gr.Tabs():
            # Tab 1: Stage 1 - Analysis & Evaluation
            with gr.TabItem("🔍 Stage 1: API Analysis & Evaluation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>📊 API Overview</h3>")
                        
                        # API status indicator
                        api_status_indicator = gr.HTML(
                            "<div style='padding: 1rem; border-radius: 8px; background: var(--glass-bg); border: 1px solid var(--glass-border); text-align: center;'><span class='status-warning'>⚠️ No API specification loaded</span></div>",
                            visible=True
                        )
                        
                        api_info_display = gr.Markdown(
                            "Upload an API specification or try the demo to see API details here.",
                            elem_classes=["analysis-panel"],
                        )

                        analyze_btn = gr.Button(
                            "🚀 Start Analysis",
                            variant="primary",
                            size="lg",
                            visible=False,
                        )

                        # Removed unused results_section components to improve performance

                    with gr.Column(scale=2):
                        gr.HTML("<h3>📈 Analysis Results</h3>")
                        analysis_output = gr.Markdown(
                            "Analysis results will appear here after running analysis.",
                            elem_classes=["analysis-panel"],
                        )

                        # Move spec display to a collapsible section
                        with gr.Accordion(
                            "📊 OpenAPI Specification (Reference)", open=False
                        ):
                            spec_display = gr.Code(
                                label="Specification JSON/YAML",
                                language="json",
                                visible=False,
                            )

            # Tab 2: Stage 2 - RAG Assistant
            with gr.TabItem("💬 Stage 2: AI Assistant"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML("<h3>🤖 AI-Powered API Documentation Assistant</h3>")
                        chatbot = gr.Chatbot(
                            label="🤖 AI-Powered API Documentation Assistant",
                            height=550,
                            elem_classes=["assistant-chat"],
                            show_copy_button=True,
                            show_share_button=False,
                            avatar_images=("🧑‍💻", "🤖"),
                            bubble_full_width=False,
                            layout="panel",
                        )

                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="💬 Your Question",
                                placeholder="Ask about endpoints, authentication, code examples, best practices...",
                                scale=4,
                                lines=2,
                                max_lines=4,
                                show_copy_button=True,
                            )
                            with gr.Column(scale=1):
                                model_dropdown = gr.Dropdown(
                                    label="🤖 AI Model",
                                    choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                                    value="gpt-4o-mini",
                                    info="Choose AI model (gpt-4o-mini is faster and cheaper)"
                                )
                                send_btn = gr.Button("🚀 Send", variant="primary", size="lg")
                                streaming_toggle = gr.Checkbox(
                                    label="⚡ Real-time Streaming",
                                    value=True,
                                    info="Enable for faster, progressive responses"
                                )

                    with gr.Column(scale=1):
                        # Real-time status indicator
                        with gr.Row():
                            connection_status = gr.HTML(
                                "<div class='status-success'>🟢 Backend Connected</div>",
                                visible=True
                            )
                        
                        gr.HTML("<h3>💡 Smart Questions</h3>")
                        with gr.Accordion("💡 Smart Question Suggestions", open=True):
                            gr.Dataset(
                                samples=[
                                    ["🔐 How do I authenticate with this API?"],
                                    ["🐍 Show me a Python code example"],
                                    ["📋 What's the expected data format?"],
                                    ["⚠️ How should I handle API errors?"],
                                    ["🌐 Give me a cURL example"],
                                    ["📝 What parameters are required?"],
                                    ["⚡ Show JavaScript/TypeScript example"],
                                    ["📚 API rate limits and best practices?"],
                                    ["🔍 Available endpoints and methods?"],
                                    ["🛠️ SDK recommendations and tools?"],
                                ],
                                headers=["Quick Questions"],
                                type="index",
                                components=[msg_input],
                            )

                        # Enhanced AI Features Panel with 2025 Design
                        with gr.Accordion("🤖 AI Assistant Features", open=False):
                            gr.HTML(
                                """
                            <div style="background: var(--glass-bg); backdrop-filter: blur(10px); padding: 20px; border-radius: 12px; border: 1px solid var(--glass-border);">
                                <h4 style="margin: 0 0 15px 0; color: var(--neon-accent); font-weight: 700;">🚀 2025 AI-Powered Assistant Features</h4>
                                
                                <div style="display: grid; gap: 15px; margin-bottom: 20px;">
                                    <div style="background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid #10b981;">
                                        <h5 style="margin: 0 0 8px 0; color: #10b981;">⚡ Real-time Streaming</h5>
                                        <p style="margin: 0; font-size: 0.9rem; color: #94a3b8; line-height: 1.4;">Get progressive responses as the AI thinks, reducing perceived latency and improving user experience.</p>
                                    </div>
                                    
                                    <div style="background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid #3b82f6;">
                                        <h5 style="margin: 0 0 8px 0; color: #3b82f6;">🎯 Context-Aware Responses</h5>
                                        <p style="margin: 0; font-size: 0.9rem; color: #94a3b8; line-height: 1.4;">AI understands your specific API documentation and provides tailored answers with relevant examples.</p>
                                    </div>
                                    
                                    <div style="background: rgba(245, 158, 11, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid #f59e0b;">
                                        <h5 style="margin: 0 0 8px 0; color: #f59e0b;">📚 Multi-Format Support</h5>
                                        <p style="margin: 0; font-size: 0.9rem; color: #94a3b8; line-height: 1.4;">Supports OpenAPI 3.0, Swagger, and custom API specification formats with intelligent parsing.</p>
                                    </div>
                                    
                                    <div style="background: rgba(168, 85, 247, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid #a855f7;">
                                        <h5 style="margin: 0 0 8px 0; color: #a855f7;">🔍 Smart Code Examples</h5>
                                        <p style="margin: 0; font-size: 0.9rem; color: #94a3b8; line-height: 1.4;">Generates Python, JavaScript, cURL, and other language examples based on your API specification.</p>
                                    </div>
                                </div>
                                
                                <div style="background: rgba(0, 245, 255, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(0, 245, 255, 0.2);">
                                    <h5 style="margin: 0 0 10px 0; color: var(--neon-accent);">💡 Pro Tips for Better Results</h5>
                                    <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem; color: #94a3b8; line-height: 1.6;">
                                        <li>Be specific about the programming language you want examples in</li>
                                        <li>Ask for complete workflows, not just individual endpoint calls</li>
                                        <li>Request error handling patterns and best practices</li>
                                        <li>Inquire about authentication flows and security considerations</li>
                                    </ul>
                                </div>
                            </div>
                            """
                            )

            # Tab 3: RAG Pipeline Study
            with gr.TabItem("🔬 RAG Pipeline Study"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML("<h3>🧬 RAG v2 Pipeline Demonstration & Study</h3>")
                        gr.Markdown(
                            """
                        **Educational Interface**: This tab demonstrates the complete RAG v2 pipeline for learning and studying purposes.
                        
                        **What you'll see:**
                        - Real-time execution of all 7 RAG v2 components
                        - Performance metrics and evaluation scores
                        - Component-by-component breakdown
                        - Visual pipeline flow visualization
                        """
                        )

                        with gr.Row():
                            run_pipeline_btn = gr.Button(
                                "🚀 Run RAG v2 Pipeline Demo",
                                variant="primary",
                                size="lg",
                            )
                            clear_pipeline_btn = gr.Button(
                                "🔄 Clear Results", variant="secondary"
                            )

                        pipeline_output = gr.Code(
                            label="RAG Pipeline Execution Log",
                            language="shell",
                            value="Click 'Run RAG v2 Pipeline Demo' to see the complete pipeline in action...",
                            lines=25,
                            elem_classes=["analysis-panel"],
                        )

                    with gr.Column(scale=1):
                        gr.HTML("<h3>📊 Pipeline Components</h3>")
                        gr.Markdown(
                            """
                        **1. 📝 Chunking Strategy**
                        - LangChain RecursiveJsonSplitter
                        - Semantic chunking for OpenAPI specs
                        - Fallback mechanism included
                        
                        **2. 🔍 Hybrid Search Engine**
                        - ChromaDB vector database
                        - BGE-M3 multilingual embeddings
                        - BM25 keyword + semantic search
                        
                        **3. 🎯 Context Assembler**
                        - Anthropic's contextual enrichment
                        - Smart deduplication
                        - Token-aware prioritization
                        
                        **4. 💾 Semantic Cache**
                        - Cosine similarity caching
                        - Performance optimization
                        - Cache hit/miss tracking
                        
                        **5. ⚡ Cache Layer**
                        - Redis backend support
                        - Memory fallback
                        - Efficient key management
                        
                        **6. 🧪 DeepEval Enhanced**
                        - RAG Triad evaluation
                        - Answer/Faithfulness/Context relevancy
                        - Mock evaluation for demo
                        
                        **7. 📈 Performance Monitor**
                        - Real-time metrics
                        - System health monitoring
                        - Time-bucketed statistics
                        """
                        )

                        gr.HTML(
                            """
                        <div style="margin-top: 20px; padding: 15px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border-left: 4px solid #3b82f6;">
                            <h4 style="margin: 0 0 10px 0; color: #3b82f6;">💡 Learning Objectives</h4>
                            <ul style="margin: 0; padding-left: 20px; font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
                                <li>Understand RAG v2 architecture</li>
                                <li>See real production-grade code in action</li>
                                <li>Learn about evaluation metrics</li>
                                <li>Observe performance characteristics</li>
                                <li>Study component interactions</li>
                            </ul>
                        </div>
                        """
                        )

        # Enhanced footer with 2025 design
        gr.HTML(
            f"""
        <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: var(--glass-bg); backdrop-filter: blur(10px); border-radius: 12px; border: 1px solid var(--glass-border);">
            <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;">
                <a href="{API_BASE_URL}/docs" target="_blank" style="color: var(--neon-accent); text-decoration: none; font-weight: 600; transition: all 0.2s;">🔗 API Documentation</a>
                <span style="color: #64748b;">•</span>
                <span style="color: #94a3b8; font-weight: 500;">📱 Mobile Optimized</span>
                <span style="color: #64748b;">•</span>
                <span style="color: #94a3b8; font-weight: 500;">♿ Accessibility Ready</span>
                <span style="color: #64748b;">•</span>
                <span style="color: #94a3b8; font-weight: 500;">⚡ Real-time Streaming</span>
            </div>
            <div style="color: #64748b; font-size: 0.85rem; line-height: 1.5;">
                <p style="margin: 0;">🚀 <strong>Workflow:</strong> Upload API spec → Set OpenAI key → Analyze quality → Ask AI questions → Study RAG pipeline</p>
                <p style="margin: 0.5rem 0 0 0;">🎯 <strong>2025 Features:</strong> Glassmorphism UI • Streaming responses • Mobile-first design • Enhanced accessibility</p>
            </div>
        </div>
        """
        )

        # Event handlers

        def update_api_status_on_load(file):
            """Update API status indicator when spec is loaded"""
            if file:
                return gr.update(value="<div style='padding: 1rem; border-radius: 8px; background: var(--glass-bg); border: 1px solid var(--success-glow); text-align: center;'><span class='status-success'>✅ API specification loaded and ready</span></div>")
            return gr.update(value="<div style='padding: 1rem; border-radius: 8px; background: var(--glass-bg); border: 1px solid var(--glass-border); text-align: center;'><span class='status-warning'>⚠️ No API specification loaded</span></div>")

        file_input.change(
            fn=load_api_spec,
            inputs=[file_input],
            outputs=[spec_display, api_info_display, status_display],
        ).then(
            fn=update_api_status_on_load,
            inputs=[file_input],
            outputs=[api_status_indicator]
        ).then(fn=lambda: gr.update(visible=True), outputs=[analyze_btn])

        demo_btn.click(
            fn=load_demo, outputs=[spec_display, api_info_display, status_display]
        ).then(
            fn=lambda: gr.update(value="<div style='padding: 1rem; border-radius: 8px; background: var(--glass-bg); border: 1px solid var(--success-glow); text-align: center;'><span class='status-success'>✅ Demo API loaded and ready</span></div>"),
            outputs=[api_status_indicator]
        ).then(fn=lambda: gr.update(visible=True), outputs=[analyze_btn])

        def start_streaming_analysis():
            """Start API analysis with proper async handling"""
            if not current_spec:
                return "❌ Please upload an API specification first"

            try:
                import asyncio
                import concurrent.futures

                async def run_analysis():
                    """Run the actual analysis"""
                    result = await start_analysis()
                    # start_analysis returns (analysis, status, metadata)
                    if isinstance(result, tuple) and len(result) >= 1:
                        return result[0]  # Return the analysis text
                    return str(result)

                # Safely run async function without interfering with existing event loops
                try:
                    # Check if we're already in an event loop
                    asyncio.get_running_loop()
                    # We're in an event loop, use ThreadPoolExecutor to run in separate thread
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, run_analysis())
                        return future.result(timeout=120)
                except RuntimeError:
                    # No event loop running, safe to use asyncio.run
                    return asyncio.run(run_analysis())

            except concurrent.futures.TimeoutError:
                logger.error("Analysis timed out after 120 seconds")
                return "❌ Analysis timed out. Please try again."
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                return f"❌ Analysis failed: {str(e)}"

        analyze_btn.click(fn=start_streaming_analysis, outputs=[analysis_output])

        # Modern event handling with enhanced chat
        def handle_chat_submission(message, history, use_streaming):
            """Handle chat submission with enhanced formatting"""
            # Always use enhanced chat (streaming functionality will be added in future)
            return chat_with_api(message, history)

        send_btn.click(
            fn=lambda msg, hist, streaming: handle_chat_submission(msg, hist, streaming),
            inputs=[msg_input, chatbot, streaming_toggle],
            outputs=[chatbot, msg_input]
        )

        msg_input.submit(
            fn=lambda msg, hist, streaming: handle_chat_submission(msg, hist, streaming),
            inputs=[msg_input, chatbot, streaming_toggle],
            outputs=[chatbot, msg_input]
        )

        # Tab 4: Multi-Agent Analysis with Streaming - Enhanced Layout
        with gr.TabItem("🤝 Stage 3: Multi-Agent Analysis", elem_id="multiagent-tab"):
            # Header section with improved styling
            gr.HTML("""
                <div class="analysis-header">
                    <h2 style="text-align: center; margin-bottom: 1rem; color: var(--neon-accent); text-shadow: 0 0 10px var(--neon-accent);">
                        🤝 Multi-Agent API Analysis
                    </h2>
                    <p style="text-align: center; font-size: 1.1rem; color: #94a3b8; margin-bottom: 2rem;">
                        Experience specialized AI agents working collaboratively with real-time streaming
                    </p>
                </div>
            """)

            # Action buttons row with better spacing
            with gr.Row(elem_classes=["action-buttons-row"]):
                multiagent_analyze_btn = gr.Button(
                    "🚀 Start Multi-Agent Analysis",
                    variant="primary",
                    size="lg",
                    scale=2,
                    elem_classes=["primary-action-btn"],
                    elem_id="start-multiagent-btn"
                )
                multiagent_clear_btn = gr.Button(
                    "🗑️ Clear Results",
                    variant="secondary",
                    scale=1,
                    elem_classes=["secondary-action-btn"]
                )
                # Spacer for better balance (scale not supported in Gradio 5.x for HTML)

            # Features showcase with modern styling
            with gr.Row(elem_classes=["features-showcase"]):
                gr.HTML("""
                    <div class="feature-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
                        <div class="feature-card">
                            <div class="feature-icon">⚡</div>
                            <h4>Real-time Updates</h4>
                            <p>See each agent working live</p>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">👥</div>
                            <h4>4 Specialized Agents</h4>
                            <p>Security, Docs, Performance, Integration</p>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">📊</div>
                            <h4>Progressive Results</h4>
                            <p>Analysis chunks as they complete</p>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">📋</div>
                            <h4>Consolidated Report</h4>
                            <p>Expert insights combined</p>
                        </div>
                    </div>
                """)

            # Progress indicators with enhanced styling
            with gr.Row(elem_classes=["progress-section"]):
                with gr.Column():
                    gr.HTML("""
                        <h3 style="color: var(--neon-accent); margin-bottom: 1rem;">
                            📊 Agent Progress Dashboard
                        </h3>
                    """)
                    agent_status = gr.HTML(
                        value="""
                        <div class="agent-progress-panel">
                            <div class="agent-status-item">
                                <div class="agent-icon">🔒</div>
                                <div class="agent-info">
                                    <strong>SecurityAnalyst</strong>
                                    <span class="status-indicator" id="sec-status">⏳ Waiting</span>
                                </div>
                            </div>
                            <div class="agent-status-item">
                                <div class="agent-icon">📝</div>
                                <div class="agent-info">
                                    <strong>DocumentationReviewer</strong>
                                    <span class="status-indicator" id="doc-status">⏳ Waiting</span>
                                </div>
                            </div>
                            <div class="agent-status-item">
                                <div class="agent-icon">⚡</div>
                                <div class="agent-info">
                                    <strong>PerformanceEngineer</strong>
                                    <span class="status-indicator" id="perf-status">⏳ Waiting</span>
                                </div>
                            </div>
                            <div class="agent-status-item">
                                <div class="agent-icon">🔗</div>
                                <div class="agent-info">
                                    <strong>IntegrationSpecialist</strong>
                                    <span class="status-indicator" id="int-status">⏳ Waiting</span>
                                </div>
                            </div>
                        </div>
                        """,
                        elem_classes=["agent-dashboard"]
                    )

            # Results section with full width and better styling
            with gr.Row(elem_classes=["results-section"]):
                with gr.Column():
                    gr.HTML("""
                        <h3 style="color: var(--neon-accent); margin-bottom: 1rem;">
                            🔄 Real-Time Analysis Stream
                        </h3>
                    """)
                    multiagent_output = gr.Markdown(
                        value="Click **'Start Multi-Agent Analysis'** to begin collaborative analysis with real-time streaming...",
                        elem_classes=["analysis-output-panel"],
                        elem_id="multiagent-results"
                    )

        # RAG Pipeline Study tab event handlers
        run_pipeline_btn.click(fn=run_rag_pipeline_demo, outputs=[pipeline_output])

        clear_pipeline_btn.click(fn=clear_pipeline_output, outputs=[pipeline_output])

        # Multi-Agent Analysis tab event handlers
        def multiagent_analyze_wrapper():
            """Wrapper to use global current_spec for multi-agent analysis with real-time dashboard"""
            global current_spec
            if current_spec:
                # Use the new generator-based function that yields both outputs
                for output, dashboard in multiagent_analyze_api_with_dashboard(current_spec):
                    yield output, dashboard
            else:
                yield "❌ Please upload an OpenAPI specification first using the file upload above.", generate_dashboard_html()

        def clear_multiagent_outputs():
            """Clear both analysis output and dashboard"""
            return "", generate_dashboard_html()

        multiagent_analyze_btn.click(
            fn=multiagent_analyze_wrapper,
            outputs=[multiagent_output, agent_status]
        )

        multiagent_clear_btn.click(
            fn=clear_multiagent_outputs,
            outputs=[multiagent_output, agent_status]
        )

        return app


if __name__ == "__main__":
    logger.info("=== APISage Complete Application Starting ===")
    logger.info("Application: APISage Complete (Stage 1 + Stage 2)")
    logger.info("Server: http://127.0.0.1:7860")
    logger.info("Features: API Analysis & Evaluation + RAG Assistant")
    logger.info(f"Backend API: {API_BASE_URL}")
    logger.info("Log file: apisage_complete.log")

    print("🚀 Launching APISage Complete Application...")
    print("🎯 Stage 1: API Analysis & Evaluation")
    print("🤖 Stage 2: AI Assistant (RAG)")
    print("📍 Access: http://localhost:7860")

    try:
        logger.info("Creating complete interface with both analysis and assistant tabs")
        app = create_complete_interface()

        logger.info("Enabling queue for generator functions")
        app.queue()

        logger.info("Launching Gradio application")
        app.launch(
            server_name="127.0.0.1", server_port=7860, inbrowser=False, quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to launch complete application: {str(e)}")
        print(f"❌ Error launching application: {str(e)}")
        raise
    finally:
        logger.info("=== APISage Complete Application Shutdown ===")



def main():
    """CLI entry point for the Gradio UI."""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("=== APISage Complete Application Starting ===")
    logger.info("Application: APISage Complete (Stage 1 + Stage 2)")
    logger.info("Server: http://127.0.0.1:7860")
    logger.info("Features: API Analysis & Evaluation + RAG Assistant")
    logger.info("Backend API: http://localhost:8080")
    logger.info("Log file: apisage_complete.log")

    print("🚀 Launching APISage Complete Application...")
    print("🎯 Stage 1: API Analysis & Evaluation")
    print("🤖 Stage 2: AI Assistant (RAG)")
    print("📍 Access: http://localhost:7860")

    try:
        logger.info("Creating complete interface with both analysis and assistant tabs")
        app = create_complete_interface()

        logger.info("Enabling queue for generator functions")
        app.queue()

        logger.info("Launching Gradio application")
        app.launch(
            server_name="127.0.0.1", server_port=7860, inbrowser=False, quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to launch complete application: {str(e)}")
        print(f"❌ Error launching application: {str(e)}")
        raise
    finally:
        logger.info("=== APISage Complete Application Shutdown ===")

