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
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('apisage_complete.log')
    ]
)

logger = logging.getLogger(__name__)

# Global state
current_spec = None

# Dark theme CSS with optimized spacing
CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    color: #f8fafc !important;
    max-width: 100% !important;
}

.hero-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    color: white !important;
    text-align: center;
}

.hero-title {
    font-size: 2rem !important;
    font-weight: bold;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 1rem !important;
    opacity: 0.9;
}

.analysis-panel {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin: 0.5rem 0 !important;
}

.assistant-chat {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
}

/* Optimize column spacing and remove excess margins */
.gradio-column {
    padding: 0.5rem !important;
    margin: 0 !important;
}

.gradio-row {
    margin: 0.5rem 0 !important;
    gap: 0.5rem !important;
}

/* Make better use of horizontal space */
.gradio-blocks > div {
    max-width: none !important;
}

/* Reduce excessive spacing in components */
.gradio-textbox, .gradio-file, .gradio-button {
    margin: 0.3rem 0 !important;
}

/* Compact headers */
.gradio-html h3, .gradio-html h2 {
    margin: 0.5rem 0 !important;
}
"""

def load_api_spec(file):
    """Load and parse API specification with logging"""
    global current_spec
    
    logger.info(f"File upload attempt: {file.name if file else 'No file'}")
    
    if not file:
        logger.warning("No file provided for upload")
        return None, "Upload an API specification to get started", ""
    
    try:
        logger.info(f"Reading file: {file.name}, size: {os.path.getsize(file.name)} bytes")
        
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
            
        logger.info(f"File content length: {len(content)} characters")
        
        if file.name.endswith(('.yaml', '.yml')):
            logger.info("Parsing as YAML file")
            spec = yaml.safe_load(content)
        else:
            logger.info("Parsing as JSON file")
            spec = json.loads(content)
        
        logger.info(f"Parsed specification keys: {list(spec.keys()) if isinstance(spec, dict) else 'Not a dict'}")
        
        current_spec = spec
        
        # Handle both OpenAPI 3.0 format and custom format
        if 'openapi' in spec or 'swagger' in spec:
            # Standard OpenAPI/Swagger format
            info = spec.get('info', {})
            servers = spec.get('servers', [])
            paths = spec.get('paths', {})
            
            api_title = info.get('title', 'API Documentation')
            api_version = info.get('version', 'Unknown')
            api_description = info.get('description', 'No description provided')
            base_url = servers[0].get('url', 'Not specified') if servers else 'Not specified'
            endpoint_count = len(paths)
        else:
            # Custom format (like your JSON)
            api_title = spec.get('api_name', spec.get('title', 'API Documentation'))
            api_version = spec.get('version', 'Unknown')
            api_description = spec.get('description', 'No description provided')
            base_url = spec.get('base_url', 'Not specified')
            # Handle custom 'endpoints' array
            endpoints_data = spec.get('endpoints', [])
            endpoint_count = len(endpoints_data)
            
            # Convert custom format to OpenAPI-like structure for processing
            paths = {}
            for endpoint in endpoints_data:
                path = endpoint.get('path', '/')
                method = endpoint.get('method', 'get').lower()
                if path not in paths:
                    paths[path] = {}
                paths[path][method] = {
                    'summary': endpoint.get('description', 'No description'),
                    'description': endpoint.get('description', 'No description')
                }
        
        logger.info(f"Building API info display - Title: {api_title}, Version: {api_version}, Endpoints: {endpoint_count}")
        
        api_info = f"""## 📋 **{api_title}**

**Version**: {api_version}  
**Base URL**: {base_url}  
**Description**: {api_description}

### 🎯 **Available Endpoints** ({endpoint_count} total)
"""
        
        # Add endpoint details
        methods_found = 0
        for path, methods in paths.items():
            api_info += f"\n**`{path}`**\n"
            if isinstance(methods, dict):
                for method, details in methods.items():
                    methods_found += 1
                    if isinstance(details, dict):
                        summary = details.get('summary', details.get('description', 'No summary'))
                        api_info += f"- **{method.upper()}** - {summary}\n"
        
        logger.info(f"Processed {methods_found} HTTP methods across {endpoint_count} endpoints")
        
        # Add additional info for both formats
        if 'openapi' in spec or 'swagger' in spec:
            # OpenAPI format contact info
            spec_info = spec.get('info', {})
            if 'contact' in spec_info:
                contact = spec_info['contact']
                api_info += f"\n### 📧 **Contact Information**\n"
                if contact.get('name'):
                    api_info += f"**Name**: {contact['name']}\n"
                if contact.get('email'):
                    api_info += f"**Email**: {contact['email']}\n"
                if contact.get('url'):
                    api_info += f"**URL**: {contact['url']}\n"
        else:
            # Custom format additional info
            if spec.get('authentication'):
                auth_info = spec['authentication']
                api_info += f"\n### 🔐 **Authentication**\n"
                api_info += f"**Type**: {auth_info.get('type', 'Not specified')}\n"
                if auth_info.get('description'):
                    api_info += f"**Details**: {auth_info['description']}\n"
            
            if spec.get('rate_limiting'):
                api_info += f"\n### ⚡ **Rate Limiting**\n"
                api_info += f"{spec['rate_limiting']}\n"
        
        spec_json = json.dumps(spec, indent=2)
        status = "✅ API specification loaded successfully"
        
        return spec_json, api_info, status
        
    except Exception as e:
        logger.error(f"Error loading specification: {str(e)}")
        return None, f"❌ **Error loading specification**: {str(e)}", "❌ Failed to load API specification"

def load_demo():
    """Load demo Pet Store API with logging"""
    global current_spec
    
    logger.info("Loading Pet Store demo API specification")
    
    demo_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Pet Store API Demo",
            "version": "1.0.0",
            "description": "A sample Pet Store API for demonstration"
        },
        "servers": [{"url": "https://petstore.swagger.io/v2"}],
        "paths": {
            "/pets": {
                "get": {"summary": "List all pets"},
                "post": {"summary": "Create a new pet"}
            },
            "/pets/{petId}": {
                "get": {"summary": "Get pet by ID"},
                "delete": {"summary": "Delete a pet"}
            }
        }
    }
    
    current_spec = demo_spec
    
    # Log demo spec details
    demo_endpoints = len(demo_spec.get('paths', {}))
    demo_methods = sum(len(methods) for methods in demo_spec.get('paths', {}).values())
    logger.info(f"Demo API loaded - {demo_endpoints} endpoints with {demo_methods} HTTP methods total")
    
    # Generate display info  
    api_info = """## 📋 **Pet Store API Demo**

**Version**: 1.0.0  
**Base URL**: https://petstore.swagger.io/v2  
**Description**: A sample Pet Store API for demonstration

### 🎯 **Available Endpoints** (2 total)

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

async def set_api_key(api_key):
    """Set OpenAI API key in backend"""
    if not api_key or not api_key.startswith('sk-'):
        return "❌ Invalid API key format. Must start with 'sk-'"
    
    try:
        logger.info("Setting OpenAI API key in backend")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8080/set-api-key",
                json={"api_key": api_key},
                timeout=10.0
            )
            
            if response.status_code == 200:
                logger.info("API key set successfully")
                return "✅ API key set successfully - Ready for analysis"
            else:
                error_msg = f"Failed to set API key (Status: {response.status_code})"
                logger.error(error_msg)
                return f"❌ {error_msg}"
                
    except Exception as e:
        error_msg = f"Error setting API key: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

def enhance_analysis_formatting(content):
    """Rich, professional formatting with enhanced content density"""
    if not content or len(content.strip()) < 10:
        return content
    
    enhanced_content = content
    
    # 1. Format Overall Score with rich visual elements
    overall_score_pattern = r'\*\*Overall Score:\*\* (\d+)/100'
    def format_overall_score(match):
        score = int(match.group(1))
        status_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        status_text = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
        
        # Create visual progress bar
        progress_blocks = "█" * (score // 10) + "░" * (10 - score // 10)
        
        # Add quality indicators
        indicators = []
        if score >= 90:
            indicators = ["🏆 Premium Quality", "✨ Production Ready", "🎯 Best Practices"]
        elif score >= 80:
            indicators = ["✅ High Quality", "🚀 Well Designed", "📝 Good Documentation"]  
        elif score >= 60:
            indicators = ["⚠️ Adequate", "🔧 Needs Refinement", "📋 Some Issues"]
        else:
            indicators = ["❌ Poor Quality", "🚨 Major Issues", "⚡ Needs Work"]

        return f"""# 📊 API Quality Analysis Report

## {status_emoji} Overall Score: {score}/100 - {status_text}

**Visual Progress:** `{progress_blocks}` ({score}%)

### Quality Indicators:
{chr(10).join(f"• {indicator}" for indicator in indicators)}

### Analysis Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

---
"""
    
    enhanced_content = re.sub(overall_score_pattern, format_overall_score, enhanced_content)
    
    # 2. Format detailed score breakdowns with rich details
    score_patterns = {
        r'- \*\*(\w+):\*\* (\d+)/100 - ([^\n]+)': lambda m: format_rich_score_item(m.group(1), int(m.group(2)), m.group(3)),
        r'• \*\*(\w+):\*\* (\d+)/100 - ([^\n]+)': lambda m: format_rich_score_item(m.group(1), int(m.group(2)), m.group(3)),
        r'◦ \*\*(\w+):\*\* (\d+)/100 - ([^\n]+)': lambda m: format_rich_score_item(m.group(1), int(m.group(2)), m.group(3)),
    }
    
    def format_rich_score_item(category, score, description):
        icons = {
            "Completeness": "🔧", "Documentation": "📚", "Security": "🔒", 
            "Usability": "👥", "Standards": "📏", "Performance": "⚡",
            "Compliance": "📏"
        }
        
        impact_levels = {
            "Completeness": "Foundation",
            "Documentation": "Developer Experience", 
            "Security": "Trust & Safety",
            "Usability": "Adoption",
            "Standards": "Maintainability",
            "Performance": "Scalability",
            "Compliance": "Reliability"
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
### {icon} {category}: {score}/100 {status} (Grade: {grade})

**Impact Area:** {impact}  
**Progress:** `{bar}` {score}%  
**Analysis:** {description}

"""
    
    for pattern, formatter in score_patterns.items():
        enhanced_content = re.sub(pattern, formatter, enhanced_content)
    
    # 3. Format critical issues with enhanced structure
    def format_critical_issues(content):
        # Format numbered critical issues
        issue_pattern = r'^(\d+\.\s+\*\*Issue:\*\*\s+.+)$'
        def format_numbered_issue(match):
            issue_text = match.group(1)
            issue_number = re.match(r'^(\d+)\.', issue_text).group(1)
            clean_text = re.sub(r'\d+\.\s+\*\*Issue:\*\*\s+', '', issue_text)
            
            priority_labels = ["🔥 Critical", "⚠️ High", "📋 Medium", "💡 Low"]
            priority = priority_labels[min(int(issue_number)-1, 3)] if issue_number.isdigit() else "📋 Issue"
            
            return f"""
#### {priority} - Issue #{issue_number}: {clean_text}

**Severity:** Priority {issue_number}  
**Category:** API Design Flaw
"""
        
        content = re.sub(issue_pattern, format_numbered_issue, content, flags=re.MULTILINE)
        
        # Format sub-bullet details with enhanced styling
        sub_bullet_pattern = r'^\s*-\s+\*\*([^*]+):\*\*\s+(.+)$'
        def format_issue_details(match):
            label = match.group(1)
            detail = match.group(2)
            
            label_icons = {
                "Location": "📍", "Impact": "💥", "Fix": "🔧", 
                "Priority": "⭐", "Reason": "💭", "Implementation": "🛠️"
            }
            
            icon = label_icons.get(label, "•")
            return f"**{icon} {label}:** {detail}"
        
        content = re.sub(sub_bullet_pattern, format_issue_details, content, flags=re.MULTILINE)
        return content
    
    enhanced_content = format_critical_issues(enhanced_content)
    
    # 4. Add summary statistics after executive summary
    def add_analysis_stats(content):
        # Count various elements for statistics
        issue_count = len(re.findall(r'#### .* - Issue #\d+', content))
        recommendation_count = len(re.findall(r'\*\*Fix:\*\*', content))
        
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
            content = content.replace("## 🎯 Executive Summary", stats_section + "## 🎯 Executive Summary")
        else:
            # Insert near the beginning
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('## ') and '🎯' not in line:
                    lines.insert(i, stats_section)
                    break
            content = '\n'.join(lines)
        
        return content
    
    enhanced_content = add_analysis_stats(enhanced_content)
    
    # 5. Format main section headers with better styling
    header_replacements = {
        r'### 🎯 Executive Summary': '## 🎯 Executive Summary\n*Key findings and overall assessment*',
        r'### 🚨 Critical Issues \(Priority Order\)': '## 🚨 Critical Issues (Priority Order)\n*Issues requiring immediate attention*', 
        r'### 🔍 Detailed Analysis': '## 🔍 Detailed Analysis\n*Comprehensive evaluation results*',
        r'### (.+)': r'## \1'
    }
    
    for pattern, replacement in header_replacements.items():
        enhanced_content = re.sub(pattern, replacement, enhanced_content)
    
    # 6. Format code blocks with language hints
    enhanced_content = re.sub(
        r'```(yaml|json|javascript|python)?\n(.*?)\n```',
        lambda m: f"```{m.group(1) or 'yaml'}\n{m.group(2)}\n```\n*Example {m.group(1) or 'configuration'} implementation*",
        enhanced_content,
        flags=re.DOTALL
    )
    
    # 7. Add visual separators and improve spacing
    enhanced_content = re.sub(r'\n---\n', '\n\n---\n\n', enhanced_content)
    enhanced_content = re.sub(r'\n{4,}', '\n\n', enhanced_content)
    enhanced_content = re.sub(r'\*\*([^*]+):\*\*', r'**\1:**', enhanced_content)
    
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

async def start_analysis_streaming():
    """Start streaming API analysis using backend"""
    if not current_spec:
        yield "❌ Please upload an API specification first"
        return
    
    try:
        logger.info("Starting streaming API analysis via backend")
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for streaming
            async with client.stream(
                "POST",
                "http://localhost:8080/analyze-stream",
                json={
                    "openapi_spec": current_spec,
                    "focus_areas": ["security", "performance", "documentation", "completeness", "standards"]
                }
            ) as response:
                
                if response.status_code != 200:
                    if response.status_code == 503:
                        yield "❌ Analysis unavailable - Please set your OpenAI API key first"
                    else:
                        yield f"❌ Analysis failed with status {response.status_code}"
                    return
                
                buffer = ""
                analysis_content = ""
                
                async for chunk in response.aiter_text():
                    buffer += chunk
                    
                    while "\n\n" in buffer:
                        line, buffer = buffer.split("\n\n", 1)
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            if data_str == "[DONE]":
                                logger.info("Streaming analysis completed successfully")
                                return
                            
                            try:
                                import json
                                data = json.loads(data_str)
                                
                                if "error" in data:
                                    yield f"❌ Error: {data['error']}"
                                    return
                                elif "status" in data:
                                    yield f"🔄 {data['message']}"
                                elif "content" in data:
                                    analysis_content += data["content"]
                                    # Apply enhanced formatting to the content
                                    enhanced_content = enhance_analysis_formatting(analysis_content)
                                    yield enhanced_content
                                    
                            except json.JSONDecodeError:
                                continue  # Skip malformed JSON
                
    except Exception as e:
        error_msg = f"❌ Streaming analysis error: {str(e)}"
        logger.error(error_msg)
        yield error_msg

async def start_analysis():
    """Start API analysis (non-streaming fallback)"""
    if not current_spec:
        return "❌ Please upload an API specification first", "", None
    
    try:
        logger.info("Starting API analysis via backend")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8080/analyze",
                json={
                    "openapi_spec": current_spec,
                    "focus_areas": ["security", "performance", "documentation", "completeness", "standards"]
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get("analysis", "Analysis completed successfully")
                # Apply enhanced formatting to the analysis
                enhanced_analysis = enhance_analysis_formatting(analysis)
                logger.info("Analysis completed successfully")
                return enhanced_analysis, "Analysis completed", None
            elif response.status_code == 503:
                error_msg = "❌ Analysis unavailable - Please set your OpenAI API key first"
                logger.error(error_msg)
                return error_msg, "", None
            else:
                error_msg = f"Analysis failed with status {response.status_code}"
                logger.error(error_msg)
                return error_msg, "", None
                
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        logger.error(error_msg)
        return error_msg, "", None

def chat_with_api(message, history):
    """Chat with the API using RAG backend with comprehensive logging"""
    logger.info(f"RAG chat request received - Message length: {len(message.strip())} chars")
    
    if not message.strip():
        logger.warning("Empty message received in chat")
        return history, ""
    
    if not current_spec:
        logger.warning("Chat attempted without API specification loaded")
        history.append([message, "Please upload an API specification first to enable AI assistance."])
        return history, ""
    
    # Log spec context info
    spec_title = current_spec.get('info', {}).get('title', 'Unknown API')
    spec_endpoints = len(current_spec.get('paths', {}))
    logger.info(f"Processing RAG query for {spec_title} with {spec_endpoints} endpoints")
    
    try:
        # Call enhanced RAG endpoint with DeepEval integration
        logger.info("Sending request to enhanced RAG backend at localhost:8080/rag-query-v2")
        import requests
        response = requests.post(
            "http://localhost:8080/rag-query-v2",
            json={
                "question": message,
                "openapi_spec": current_spec
            },
            timeout=30.0
        )
        
        logger.info(f"RAG backend response - Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "Sorry, I couldn't generate a response.")
            metadata = result.get("metadata", {})
            rag_triad = metadata.get("rag_triad_evaluation", {})
            performance = metadata.get("performance", {})
            
            # Create enhanced response with evaluation metrics
            enhanced_answer = answer
            
            # Add DeepEval RAG Triad scores if available
            if rag_triad:
                eval_display = f"\n\n📊 **DeepEval RAG Triad Evaluation Results:**\n"
                eval_display += "=" * 60 + "\n\n"
                
                # Answer Relevancy Score
                answer_relevancy = rag_triad.get('answer_relevancy', 0)
                eval_display += f"### 🎯 **Answer Relevancy: {answer_relevancy:.3f}**\n"
                eval_display += "**What it measures:** How well the answer addresses your specific question\n"
                eval_display += "**Formula:** `Cosine Similarity(Question Embedding, Answer Embedding)`\n"
                eval_display += "**Interpretation:** "
                if answer_relevancy >= 0.8:
                    eval_display += "✅ Excellent - Answer directly addresses the question\n"
                elif answer_relevancy >= 0.6:
                    eval_display += "✔️ Good - Answer is mostly relevant with minor deviations\n"
                elif answer_relevancy >= 0.4:
                    eval_display += "⚠️ Fair - Answer partially addresses the question\n"
                else:
                    eval_display += "❌ Poor - Answer may be off-topic or incomplete\n"
                eval_display += "\n"
                
                # Faithfulness Score
                faithfulness = rag_triad.get('faithfulness', 0)
                eval_display += f"### 🔍 **Faithfulness: {faithfulness:.3f}**\n"
                eval_display += "**What it measures:** Accuracy of the answer relative to source documents\n"
                eval_display += "**Formula:** `(Verified Claims / Total Claims)` where claims are validated against context\n"
                eval_display += "**Interpretation:** "
                if faithfulness >= 0.9:
                    eval_display += "✅ Excellent - All claims are supported by documentation\n"
                elif faithfulness >= 0.7:
                    eval_display += "✔️ Good - Most claims are accurate with minor issues\n"
                elif faithfulness >= 0.5:
                    eval_display += "⚠️ Fair - Some claims lack proper support\n"
                else:
                    eval_display += "❌ Poor - Answer contains unsupported or incorrect claims\n"
                eval_display += "\n"
                
                # Contextual Relevancy Score
                contextual_relevancy = rag_triad.get('contextual_relevancy', 0)
                eval_display += f"### 📋 **Contextual Relevancy: {contextual_relevancy:.3f}**\n"
                eval_display += "**What it measures:** Quality of retrieved documentation chunks\n"
                eval_display += "**Formula:** `(Relevant Chunks / Total Retrieved Chunks)`\n"
                eval_display += "**Interpretation:** "
                if contextual_relevancy >= 0.8:
                    eval_display += "✅ Excellent - Retrieved highly relevant documentation\n"
                elif contextual_relevancy >= 0.6:
                    eval_display += "✔️ Good - Most retrieved content is useful\n"
                elif contextual_relevancy >= 0.4:
                    eval_display += "⚠️ Fair - Mixed relevance in retrieved content\n"
                else:
                    eval_display += "❌ Poor - Retrieved content has low relevance\n"
                eval_display += "\n"
                
                # Overall Score
                overall_score = rag_triad.get('overall_score', 0)
                eval_display += f"### 🏆 **Overall Score: {overall_score:.3f}**\n"
                eval_display += "**Calculation:** Weighted average of all metrics\n"
                eval_display += f"**Formula:** `(0.4 × Answer Relevancy + 0.4 × Faithfulness + 0.2 × Context Relevancy)`\n"
                eval_display += f"**Breakdown:** `(0.4 × {answer_relevancy:.3f} + 0.4 × {faithfulness:.3f} + 0.2 × {contextual_relevancy:.3f}) = {overall_score:.3f}`\n"
                eval_display += "\n"
                
                # Confidence Score
                confidence = rag_triad.get('confidence', 0)
                eval_display += f"### 🎲 **Confidence: {confidence:.3f}**\n"
                eval_display += "**What it measures:** System's confidence in the evaluation\n"
                eval_display += "**Formula:** `min(1.0, Overall Score × Context Quality Factor × Response Coherence)`\n"
                eval_display += "**Interpretation:** "
                if confidence >= 0.8:
                    eval_display += "✅ High confidence - Reliable evaluation\n"
                elif confidence >= 0.6:
                    eval_display += "✔️ Moderate confidence - Generally trustworthy\n"
                elif confidence >= 0.4:
                    eval_display += "⚠️ Low confidence - Results may vary\n"
                else:
                    eval_display += "❌ Very low confidence - Evaluation uncertain\n"
                
                # Add performance metrics if available
                if performance:
                    eval_display += f"\n⚡ **Performance:**\n"
                    eval_display += f"Search: {performance.get('search_time_ms', 0):.1f}ms | "
                    eval_display += f"LLM: {performance.get('llm_time_ms', 0):.1f}ms | "
                    eval_display += f"Total: {performance.get('total_time_ms', 0):.1f}ms\n"
                    eval_display += f"Tokens/sec: {performance.get('tokens_per_second', 0):.1f}\n"
                
                # Add quality assessment with educational explanation
                eval_display += "\n" + "=" * 60 + "\n"
                eval_display += "## 📈 **Quality Assessment**\n"
                overall_score = rag_triad.get('overall_score', 0)
                
                if overall_score >= 0.8:
                    eval_display += f"### ✅ **Excellent Quality** (Score: {overall_score:.3f})\n"
                    eval_display += "**What this means:** The AI has provided a highly accurate, relevant, and well-supported answer.\n"
                    eval_display += "**You can:** Trust this response with high confidence for critical decisions.\n"
                elif overall_score >= 0.6:
                    eval_display += f"### ✔️ **Good Quality** (Score: {overall_score:.3f})\n"
                    eval_display += "**What this means:** The answer is generally reliable with minor areas for improvement.\n"
                    eval_display += "**You can:** Use this response confidently, but verify critical details.\n"
                elif overall_score >= 0.4:
                    eval_display += f"### ⚠️ **Fair Quality** (Score: {overall_score:.3f})\n"
                    eval_display += "**What this means:** The answer has some useful information but may lack completeness or accuracy.\n"
                    eval_display += "**You should:** Cross-reference important information and ask follow-up questions.\n"
                else:
                    eval_display += f"### ❌ **Needs Improvement** (Score: {overall_score:.3f})\n"
                    eval_display += "**What this means:** The answer may be off-topic, incomplete, or contain inaccuracies.\n"
                    eval_display += "**You should:** Rephrase your question or provide more context for better results.\n"
                
                # Add learning tips
                eval_display += "\n### 💡 **Tips for Better Results:**\n"
                if answer_relevancy < 0.7:
                    eval_display += "• **Be more specific:** Your question might be too broad or vague\n"
                if faithfulness < 0.7:
                    eval_display += "• **Check documentation:** The system may lack sufficient context\n"
                if contextual_relevancy < 0.7:
                    eval_display += "• **Provide context:** Include more details about your use case\n"
                if overall_score >= 0.7:
                    eval_display += "• **Great job!** Your question was well-formed and got quality results\n"
                    
                enhanced_answer += eval_display
            
            logger.info(f"RAG response received - Answer length: {len(answer)} chars, RAG Triad: {bool(rag_triad)}")
            history.append([message, enhanced_answer])
        else:
            error_msg = f"Error: Unable to get response from AI service (Status: {response.status_code})"
            logger.error(f"RAG backend error - Status: {response.status_code}")
            history.append([message, error_msg])
                
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"RAG request failed: {str(e)}")
        history.append([message, error_msg])
    
    return history, ""

def create_complete_interface():
    """Create the complete interface with both evaluation and assistant tabs"""
    with gr.Blocks(css=CUSTOM_CSS, title="APISage Complete - AI-Powered API Analysis & Assistant") as app:
        
        # Hero Section
        gr.HTML("""
        <div class="hero-section">
            <div class="hero-title">🚀 APISage Complete</div>
            <div class="hero-subtitle">Stage 1: API Analysis & Evaluation • Stage 2: RAG Assistant</div>
        </div>
        """)
        
        # Compact configuration section
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("### 🔑 API Configuration")
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password",
                        scale=2
                    )
                    set_key_btn = gr.Button("Set Key", variant="primary", scale=1)
            
            with gr.Column(scale=2):
                gr.HTML("### 📁 API Specification")
                with gr.Row():
                    file_input = gr.File(
                        label="Upload JSON/YAML",
                        file_types=[".json", ".yaml", ".yml"],
                        scale=2
                    )
                    demo_btn = gr.Button("🐾 Demo", variant="secondary", scale=1)
        
        with gr.Row():
            key_status = gr.Textbox(
                label="API Key Status",
                value="❌ API key not set - required for analysis",
                interactive=False,
                scale=1
            )
            status_display = gr.Textbox(
                label="Upload Status",
                value="Upload an API specification to get started",
                interactive=False,
                scale=1
            )
        
        # Main tabs
        with gr.Tabs():
            # Tab 1: Stage 1 - Analysis & Evaluation
            with gr.TabItem("🔍 Stage 1: API Analysis & Evaluation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("### 📊 API Overview")
                        api_info_display = gr.Markdown(
                            "Upload an API specification or try the demo to see API details here.",
                            elem_classes=["analysis-panel"]
                        )
                        
                        analyze_btn = gr.Button(
                            "🚀 Start Analysis",
                            variant="primary",
                            size="lg",
                            visible=False
                        )
                        
                        # Move scores and charts to sidebar for compact layout
                        results_section = gr.Column(visible=False)
                        with results_section:
                            score_display = gr.HTML("")
                            radar_chart = gr.Plot()
                    
                    with gr.Column(scale=2):
                        gr.HTML("### 📈 Analysis Results")
                        analysis_output = gr.Markdown(
                            "Analysis results will appear here after running analysis.",
                            elem_classes=["analysis-panel"]
                        )
                        
                        # Move spec display to a collapsible section
                        with gr.Accordion("📊 OpenAPI Specification (Reference)", open=False):
                            spec_display = gr.Code(
                                label="Specification JSON/YAML",
                                language="json",
                                visible=False
                            )
                    
            # Tab 2: Stage 2 - RAG Assistant
            with gr.TabItem("💬 Stage 2: AI Assistant"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML("### 🤖 AI-Powered API Documentation Assistant")
                        chatbot = gr.Chatbot(
                            label="API Documentation Assistant",
                            height=500,
                            elem_classes=["assistant-chat"]
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Your Question",
                                placeholder="Ask about endpoints, authentication, code examples...",
                                scale=4
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Column(scale=1):
                        gr.HTML("### 💡 Example Questions")
                        gr.Dataset(
                            samples=[
                                ["How do I authenticate?"],
                                ["Show Python code example"],
                                ["What's the data format?"],
                                ["How to handle errors?"],
                                ["Give me cURL example"],
                                ["Required parameters?"],
                                ["JavaScript example"]
                            ],
                            headers=["Quick Asks"],
                            type="index",
                            components=[msg_input]
                        )
                        
                        # Add DeepEval info panel
                        gr.HTML("""
                        <div style="margin-top: 20px; padding: 15px; background: rgba(100, 116, 139, 0.1); border-radius: 8px; border-left: 4px solid #64748b;">
                            <h4 style="margin: 0 0 10px 0; color: #64748b;">🧪 Educational: Understanding RAG Evaluation</h4>
                            <p style="margin: 0 0 10px 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">
                                <strong>What is RAG Triad?</strong><br>
                                A comprehensive evaluation framework that measures three critical aspects of AI-generated responses:
                            </p>
                            <ul style="margin: 0; padding-left: 20px; font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
                                <li><strong>🎯 Answer Relevancy:</strong> Semantic similarity between question and answer using cosine similarity of embeddings</li>
                                <li><strong>🔍 Faithfulness:</strong> Percentage of claims in the answer that can be verified from source documents</li>
                                <li><strong>📋 Context Relevancy:</strong> Ratio of relevant vs. irrelevant chunks in retrieved documentation</li>
                            </ul>
                            <p style="margin: 10px 0 0 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">
                                <strong>How Overall Score is Calculated:</strong><br>
                                <code style="background: rgba(0,0,0,0.2); padding: 2px 4px; border-radius: 3px;">
                                Overall = (0.4 × Answer Relevancy) + (0.4 × Faithfulness) + (0.2 × Context Relevancy)
                                </code><br>
                                <em style="font-size: 0.8rem;">Higher weights on answer quality, lower on retrieval quality</em>
                            </p>
                            <p style="margin: 10px 0 0 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">
                                <strong>Why This Matters:</strong><br>
                                • Transparency: See exactly how AI responses are evaluated<br>
                                • Trust: Understand the reliability of each answer<br>
                                • Learning: Improve your questions based on scoring feedback
                            </p>
                        </div>
                        """)
        
        # Compact footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 1rem; opacity: 0.7; font-size: 0.9rem;">
            <p>🔗 <a href="http://localhost:8080/docs" target="_blank">API Docs</a> • 
            📍 Upload spec → Analyze → Ask questions</p>
        </div>
        """)
        
        # Event handlers
        set_key_btn.click(
            fn=set_api_key,
            inputs=[api_key_input],
            outputs=[key_status]
        ).then(
            fn=lambda: "",
            outputs=[api_key_input]
        )
        
        file_input.change(
            fn=load_api_spec,
            inputs=[file_input],
            outputs=[spec_display, api_info_display, status_display]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[analyze_btn]
        )
        
        demo_btn.click(
            fn=load_demo,
            outputs=[spec_display, api_info_display, status_display]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[analyze_btn]
        )
        
        def start_streaming_analysis():
            """Wrapper to handle streaming analysis"""
            import asyncio
            
            if not current_spec:
                return "❌ Please upload an API specification first"
            
            # Initialize with starting message
            analysis_text = "🔄 Initializing analysis..."
            
            try:
                # Create event loop if not exists
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the streaming analysis
                stream_gen = start_analysis_streaming()
                
                # Collect all streaming content
                full_analysis = ""
                async def collect_stream():
                    nonlocal full_analysis
                    async for chunk in stream_gen:
                        full_analysis = chunk
                    return full_analysis
                
                # Run async function
                if loop.is_running():
                    # If loop is running, use asyncio.create_task
                    task = loop.create_task(collect_stream())
                    result = task.result() if task.done() else full_analysis
                else:
                    result = loop.run_until_complete(collect_stream())
                
                return result or "Analysis completed"
                
            except Exception as e:
                logger.error(f"Streaming analysis wrapper error: {str(e)}")
                # Fallback to regular analysis
                try:
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(start_analysis())
                    return result[0] if isinstance(result, tuple) else str(result)
                except Exception as fallback_error:
                    return f"❌ Analysis failed: {str(fallback_error)}"

        analyze_btn.click(
            fn=start_streaming_analysis,
            outputs=[analysis_output]
        )
        
        send_btn.click(
            fn=chat_with_api,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            fn=chat_with_api,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        return app

if __name__ == "__main__":
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
        
        logger.info("Launching Gradio application")
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to launch complete application: {str(e)}")
        print(f"❌ Error launching application: {str(e)}")
        raise
    finally:
        logger.info("=== APISage Complete Application Shutdown ===")