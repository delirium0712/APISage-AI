"""
Multi-Agent API Analysis Orchestrator
Coordinates specialized agents for comprehensive API specification review
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import structlog
from .llm_manager import LLMRequest, SimpleLLMManager

logger = structlog.get_logger()


@dataclass
class AgentResult:
    """Result from an individual agent analysis"""
    agent_name: str
    analysis: str
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    severity_score: int  # 1-10 scale
    confidence: float  # 0-1 scale
    execution_time_ms: float
    token_usage: Dict[str, int]


@dataclass
class OrchestrationResult:
    """Result from the entire multi-agent orchestration"""
    request_id: str
    api_spec_title: str
    total_execution_time_ms: float
    agent_results: List[AgentResult]
    consolidated_analysis: str
    priority_issues: List[Dict[str, Any]]
    overall_score: int  # 1-100 scale
    metadata: Dict[str, Any]


class BaseAgent:
    """Base class for all specialized agents"""

    def __init__(self, llm_manager: SimpleLLMManager, name: str, expertise: str):
        self.llm_manager = llm_manager
        self.name = name
        self.expertise = expertise

    async def analyze(self, api_spec: Dict[str, Any], context: str = "") -> AgentResult:
        """Analyze API specification from agent's specialized perspective"""
        start_time = time.time()

        prompt = self._create_prompt(api_spec, context)

        llm_request = LLMRequest(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.3
        )

        try:
            response = await self.llm_manager.generate(llm_request)

            if not response or not response.content:
                raise Exception(f"No response from LLM for {self.name}")

            # Parse the structured response
            result = self._parse_response(response.content)

            execution_time = (time.time() - start_time) * 1000

            return AgentResult(
                agent_name=self.name,
                analysis=result["analysis"],
                findings=result["findings"],
                recommendations=result["recommendations"],
                severity_score=result["severity_score"],
                confidence=result["confidence"],
                execution_time_ms=execution_time,
                token_usage=response.usage if response.usage else {}
            )

        except Exception as e:
            logger.error(f"Agent {self.name} failed", error=str(e))
            return AgentResult(
                agent_name=self.name,
                analysis=f"Analysis failed: {str(e)}",
                findings=[],
                recommendations=[],
                severity_score=0,
                confidence=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                token_usage={}
            )

    def _create_prompt(self, api_spec: Dict[str, Any], context: str) -> str:
        """Create specialized prompt for this agent - override in subclasses"""
        raise NotImplementedError

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response into structured format"""
        # Simple parsing - in production, use more robust JSON parsing
        try:
            # Look for structured sections in the response
            analysis = response
            findings = []
            recommendations = []
            severity_score = 5
            confidence = 0.8

            # Extract findings (look for bullet points or numbered lists)
            lines = response.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if 'finding' in line.lower() or 'issue' in line.lower():
                    if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                        findings.append({
                            "description": line,
                            "severity": "medium",
                            "category": self.expertise
                        })
                elif 'recommend' in line.lower() or 'suggest' in line.lower():
                    if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                        recommendations.append(line)
                elif 'critical' in line.lower() or 'high' in line.lower():
                    severity_score = min(severity_score + 2, 10)
                elif 'low' in line.lower() or 'minor' in line.lower():
                    severity_score = max(severity_score - 1, 1)

            return {
                "analysis": analysis,
                "findings": findings,
                "recommendations": recommendations,
                "severity_score": severity_score,
                "confidence": confidence
            }

        except Exception as e:
            logger.warning(f"Failed to parse {self.name} response", error=str(e))
            return {
                "analysis": response,
                "findings": [],
                "recommendations": [],
                "severity_score": 5,
                "confidence": 0.5
            }


class SecurityAnalyst(BaseAgent):
    """Specialized agent for security analysis"""

    def __init__(self, llm_manager: SimpleLLMManager):
        super().__init__(llm_manager, "Security Analyst", "API Security")

    def _create_prompt(self, api_spec: Dict[str, Any], context: str) -> str:
        """Create security-focused analysis prompt"""
        info = api_spec.get("info", {})
        paths = api_spec.get("paths", {})
        security = api_spec.get("security", [])
        components = api_spec.get("components", {})

        return f"""You are a Senior API Security Analyst. Perform a comprehensive security analysis of this OpenAPI specification.

API SPECIFICATION:
Title: {info.get("title", "Unknown")}
Version: {info.get("version", "Unknown")}
Endpoints: {len(paths)}
Security Schemes: {len(security)}

FULL SPEC:
{json.dumps(api_spec, indent=2)[:5000]}

SECURITY ANALYSIS REQUIREMENTS:
1. **Authentication & Authorization**
   - Evaluate security schemes (Bearer, OAuth2, API Key, etc.)
   - Check for missing authentication on sensitive endpoints
   - Assess authorization patterns and access controls

2. **Input Validation & Sanitization**
   - Review parameter validation (query, path, body)
   - Check for injection vulnerabilities (SQL, NoSQL, LDAP)
   - Assess data type enforcement and bounds checking

3. **Data Protection**
   - Identify sensitive data exposure risks
   - Check HTTPS enforcement
   - Evaluate data masking and encryption requirements

RESPONSE FORMAT:
## Security Analysis Summary
[Brief overview of security posture]

## Critical Security Findings
- Finding 1: [Specific security issue]
- Finding 2: [Another security issue]

## Security Recommendations
- Recommendation 1: [Specific action to take]
- Recommendation 2: [Another recommendation]

## Security Score: X/10
[Overall security rating with justification]

### 🔐 Security Implementation Example:
```yaml
# Add this to your OpenAPI spec
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    apiKey:
      type: apiKey
      in: header
      name: X-API-Key
security:
  - bearerAuth: []
  - apiKey: []
```

### 🛡️ Error Response Security:
```yaml
responses:
  '401':
    description: Unauthorized
    content:
      application/json:
        schema:
          type: object
          properties:
            error:
              type: string
              example: 'Invalid authentication credentials'
```

Provide specific, actionable findings with concrete YAML/JSON examples from the API specification.
"""


class DocumentationReviewer(BaseAgent):
    """Specialized agent for documentation quality review"""

    def __init__(self, llm_manager: SimpleLLMManager):
        super().__init__(llm_manager, "Documentation Reviewer", "API Documentation")

    def _create_prompt(self, api_spec: Dict[str, Any], context: str) -> str:
        """Create documentation-focused analysis prompt"""
        info = api_spec.get("info", {})
        paths = api_spec.get("paths", {})
        components = api_spec.get("components", {})

        return f"""You are a Senior Technical Writer and API Documentation Specialist. Review this OpenAPI specification for documentation quality and developer experience.

API SPECIFICATION:
Title: {info.get("title", "Unknown")}
Description: {info.get("description", "No description")}
Endpoints: {len(paths)}
Components: {len(components.get("schemas", {}))}

FULL SPEC:
{json.dumps(api_spec, indent=2)[:5000]}

DOCUMENTATION REVIEW AREAS:
1. **API Metadata Quality**
   - Evaluate title, description, version information
   - Check contact information and external documentation

2. **Endpoint Documentation**
   - Review operation summaries and descriptions
   - Check parameter documentation completeness

3. **Schema Documentation**
   - Review data model descriptions
   - Check field-level documentation

RESPONSE FORMAT:
## Documentation Quality Analysis
[Overview of documentation strengths and gaps]

## Documentation Issues Found
- Issue 1: [Specific missing documentation]
- Issue 2: [Another documentation gap]

## Developer Experience Recommendations
- Recommendation 1: [Suggestion to improve usability]
- Recommendation 2: [Another improvement]

## Documentation Score: X/10
[Overall documentation quality rating]

### 📚 Documentation Enhancement Example:
```yaml
# Enhanced endpoint documentation
paths:
  /pets/{{petId}}:
    get:
      summary: Retrieve pet details
      description: |
        Fetches detailed information about a specific pet by ID.

        **Use cases:**
        - Display pet profile page
        - Verify pet ownership
        - Check pet availability status
      operationId: getPetById
      tags:
        - Pets
      parameters:
        - name: petId
          in: path
          required: true
          description: Unique identifier of the pet
          schema:
            type: string
            format: uuid
            example: "123e4567-e89b-12d3-a456-426614174000"
      responses:
        '200':
          description: Pet details retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Pet'
              examples:
                dog:
                  summary: Example dog response
                  value:
                    id: "123e4567-e89b-12d3-a456-426614174000"
                    name: "Buddy"
                    species: "dog"
                    breed: "Golden Retriever"
        '404':
          description: Pet not found
```

### 📝 Schema Documentation Example:
```yaml
components:
  schemas:
    Pet:
      type: object
      description: Represents a pet in the system
      required:
        - id
        - name
        - species
      properties:
        id:
          type: string
          format: uuid
          description: Unique identifier for the pet
          readOnly: true
        name:
          type: string
          description: Pet's name
          minLength: 1
          maxLength: 100
          example: "Fluffy"
        species:
          type: string
          description: Type of animal
          enum: [dog, cat, bird, fish, other]
          example: "cat"
```

Provide specific examples with actual YAML improvements.
"""


class PerformanceEngineer(BaseAgent):
    """Specialized agent for performance analysis"""

    def __init__(self, llm_manager: SimpleLLMManager):
        super().__init__(llm_manager, "Performance Engineer", "API Performance")

    def _create_prompt(self, api_spec: Dict[str, Any], context: str) -> str:
        """Create performance-focused analysis prompt"""
        info = api_spec.get("info", {})
        paths = api_spec.get("paths", {})

        return f"""You are a Senior Performance Engineer specializing in API optimization. Analyze this OpenAPI specification for performance considerations and scalability concerns.

API SPECIFICATION:
Title: {info.get("title", "Unknown")}
Endpoints: {len(paths)}
Total Operations: {sum(len(methods) for methods in paths.values())}

FULL SPEC:
{json.dumps(api_spec, indent=2)[:5000]}

PERFORMANCE ANALYSIS AREAS:
1. **Endpoint Design Efficiency**
   - Evaluate REST pattern adherence
   - Check for chatty vs chunky interface patterns

2. **Response Size & Pagination**
   - Review response payload sizes
   - Check pagination implementation

3. **Caching Strategy**
   - Evaluate cacheable operations
   - Check cache header specifications

RESPONSE FORMAT:
## Performance Analysis Summary
[Overview of API performance characteristics]

## Performance Bottlenecks Identified
- Bottleneck 1: [Specific performance issue]
- Bottleneck 2: [Another performance concern]

## Optimization Recommendations
- Recommendation 1: [Performance improvement suggestion]
- Recommendation 2: [Another optimization]

## Performance Score: X/10
[Overall performance design rating]

Focus on specific, measurable performance improvements with implementation guidance.
"""


class IntegrationSpecialist(BaseAgent):
    """Specialized agent for integration and compatibility analysis"""

    def __init__(self, llm_manager: SimpleLLMManager):
        super().__init__(llm_manager, "Integration Specialist", "API Integration")

    def _create_prompt(self, api_spec: Dict[str, Any], context: str) -> str:
        """Create integration-focused analysis prompt"""
        info = api_spec.get("info", {})
        paths = api_spec.get("paths", {})
        servers = api_spec.get("servers", [])

        return f"""You are a Senior Integration Architect specializing in API ecosystem design. Analyze this OpenAPI specification for integration patterns, compatibility, and ecosystem fit.

API SPECIFICATION:
Title: {info.get("title", "Unknown")}
Version: {info.get("version", "Unknown")}
Servers: {len(servers)}
Endpoints: {len(paths)}

FULL SPEC:
{json.dumps(api_spec, indent=2)[:5000]}

INTEGRATION ANALYSIS AREAS:
1. **API Design Patterns**
   - Evaluate RESTful design adherence
   - Check resource modeling and relationships

2. **Data Format & Standards**
   - Review JSON schema design quality
   - Check data type consistency

3. **Error Handling Patterns**
   - Evaluate error response consistency
   - Check HTTP status code usage

RESPONSE FORMAT:
## Integration Analysis Summary
[Overview of integration characteristics]

## Integration Challenges Identified
- Challenge 1: [Specific integration issue]
- Challenge 2: [Another integration concern]

## Integration Recommendations
- Recommendation 1: [Better integration suggestion]
- Recommendation 2: [Another improvement]

## Integration Score: X/10
[Overall integration-friendliness rating]

Provide specific guidance on improving API integration experience and ecosystem compatibility.
"""


class AgenticOrchestrator:
    """Orchestrates multiple specialized agents for comprehensive API analysis"""

    def __init__(self, llm_manager: SimpleLLMManager):
        self.llm_manager = llm_manager
        self.agents = [
            SecurityAnalyst(llm_manager),
            DocumentationReviewer(llm_manager),
            PerformanceEngineer(llm_manager),
            IntegrationSpecialist(llm_manager)
        ]

    async def analyze(
        self,
        api_spec: Dict[str, Any],
        request_id: str,
        selected_agents: Optional[List[str]] = None
    ) -> OrchestrationResult:
        """Orchestrate multi-agent analysis of API specification"""
        start_time = time.time()

        logger.info(f"Starting multi-agent analysis", request_id=request_id)

        # Filter agents if specific ones are requested
        active_agents = self.agents
        if selected_agents:
            active_agents = [
                agent for agent in self.agents
                if agent.name in selected_agents
            ]

        # Run agents in parallel
        agent_tasks = [
            agent.analyze(api_spec)
            for agent in active_agents
        ]

        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Filter out exceptions and convert to AgentResult objects
        valid_results = []
        for i, result in enumerate(agent_results):
            if isinstance(result, Exception):
                logger.error(f"Agent {active_agents[i].name} failed", error=str(result))
                # Create failed result
                valid_results.append(AgentResult(
                    agent_name=active_agents[i].name,
                    analysis=f"Agent failed: {str(result)}",
                    findings=[],
                    recommendations=[],
                    severity_score=0,
                    confidence=0.0,
                    execution_time_ms=0,
                    token_usage={}
                ))
            else:
                valid_results.append(result)

        # Consolidate results
        consolidated_analysis = await self._consolidate_results(valid_results, api_spec)
        priority_issues = self._extract_priority_issues(valid_results)
        overall_score = self._calculate_overall_score(valid_results)

        total_time = (time.time() - start_time) * 1000

        logger.info(
            "Multi-agent analysis completed",
            request_id=request_id,
            agents_count=len(valid_results),
            total_time_ms=total_time,
            overall_score=overall_score
        )

        return OrchestrationResult(
            request_id=request_id,
            api_spec_title=api_spec.get("info", {}).get("title", "Unknown API"),
            total_execution_time_ms=total_time,
            agent_results=valid_results,
            consolidated_analysis=consolidated_analysis,
            priority_issues=priority_issues,
            overall_score=overall_score,
            metadata={
                "agents_used": [agent.name for agent in active_agents],
                "total_findings": sum(len(r.findings) for r in valid_results),
                "total_recommendations": sum(len(r.recommendations) for r in valid_results),
                "average_confidence": sum(r.confidence for r in valid_results) / len(valid_results) if valid_results else 0
            }
        )

    async def _consolidate_results(
        self,
        agent_results: List[AgentResult],
        api_spec: Dict[str, Any]
    ) -> str:
        """Create a consolidated analysis from all agent results"""

        # Create consolidation prompt
        agent_summaries = []
        for result in agent_results:
            agent_summaries.append(f"""
{result.agent_name} Analysis:
Score: {result.severity_score}/10
Confidence: {result.confidence:.1%}
Key Findings: {len(result.findings)} issues identified
Top Recommendations: {result.recommendations[:2] if result.recommendations else ['No specific recommendations']}
""")

        consolidation_prompt = f"""You are a Senior API Architect consolidating analysis from multiple specialized agents. Create a highly structured, actionable report with visual scoring and code examples.

API ANALYZED: {api_spec.get('info', {}).get('title', 'Unknown API')}

AGENT RESULTS:
{chr(10).join(agent_summaries)}

REQUIRED OUTPUT FORMAT:

# 🎯 API Analysis Report: {api_spec.get('info', {}).get('title', 'Unknown API')}

## 📊 Overall Score Card
Create a visual scorecard with scores out of 100:
- 🔒 **Security Score**: [X/100] [Grade A-F]
- 📝 **Documentation Score**: [X/100] [Grade A-F]
- ⚡ **Performance Score**: [X/100] [Grade A-F]
- 🔗 **Integration Score**: [X/100] [Grade A-F]
- **Overall API Grade**: [A-F]

## 🚨 Critical Issues (Top Priority)
List the top 5 most critical issues with severity indicators:
1. 🔴 **[High]** [Issue] - [Specific problem with impact]
2. 🟠 **[Medium]** [Issue] - [Specific problem with impact]
...

## ✅ Quick Wins (Immediate Actions)
Provide 3-5 actions that can be implemented immediately:
- [ ] [Action item with specific steps]
- [ ] [Action item with specific steps]
...

## 📋 Detailed Recommendations

### 1. Security Enhancements
**Issue**: [Specific security problem]
**Solution**: [Detailed fix with YAML/JSON example]
```yaml
# Example implementation
[Provide actual code example]
```

### 2. Documentation Improvements
**Issue**: [Specific documentation gap]
**Solution**: [Detailed fix with example]
```yaml
# Example enhancement
[Provide actual documentation example]
```

### 3. Performance Optimizations
**Issue**: [Specific performance concern]
**Solution**: [Detailed optimization with example]
```yaml
# Example optimization
[Provide actual code example]
```

## 🎯 Action Plan Priority Matrix

| Priority | Category | Action | Effort | Impact |
|----------|----------|--------|--------|--------|
| 🔴 High | Security | [Specific action] | Low/Medium/High | High |
| 🟠 Medium | Docs | [Specific action] | Low/Medium/High | Medium |
| 🟢 Low | Performance | [Specific action] | Low/Medium/High | Low |

## 📈 Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
- Fix authentication vulnerabilities
- Add missing error responses
- Update API documentation

### Phase 2: Enhancements (Week 2-3)
- Implement caching strategy
- Add comprehensive examples
- Optimize response payloads

### Phase 3: Polish (Week 4)
- Add rate limiting
- Implement monitoring
- Complete test coverage

## ✨ Summary
Provide a brief, executive-level summary with:
- Current state assessment
- Key risks and opportunities
- Recommended next steps
- Expected outcomes after improvements

Use emojis, visual indicators, and formatted tables to make the report engaging and scannable.
"""

        try:
            llm_request = LLMRequest(
                prompt=consolidation_prompt,
                max_tokens=1500,
                temperature=0.2
            )

            response = await self.llm_manager.generate(llm_request)

            if response and response.content:
                return response.content
            else:
                return self._create_fallback_consolidation(agent_results)

        except Exception as e:
            logger.error("Consolidation failed", error=str(e))
            return self._create_fallback_consolidation(agent_results)

    def _create_fallback_consolidation(self, agent_results: List[AgentResult]) -> str:
        """Create an enhanced fallback consolidation with visual formatting"""
        # Calculate aggregate scores
        security_score = next((r.severity_score * 10 for r in agent_results if 'Security' in r.agent_name), 50)
        doc_score = next((r.severity_score * 10 for r in agent_results if 'Documentation' in r.agent_name), 50)
        perf_score = next((r.severity_score * 10 for r in agent_results if 'Performance' in r.agent_name), 50)
        int_score = next((r.severity_score * 10 for r in agent_results if 'Integration' in r.agent_name), 50)

        avg_score = (security_score + doc_score + perf_score + int_score) / 4
        grade = 'A' if avg_score >= 90 else 'B' if avg_score >= 80 else 'C' if avg_score >= 70 else 'D' if avg_score >= 60 else 'F'

        summary = f"""# 🎯 Multi-Agent API Analysis Report

## 📊 Overall Score Card

| Category | Score | Grade | Status |
|----------|-------|-------|--------|
| 🔒 Security | {100-security_score}/100 | {self._get_grade(100-security_score)} | {self._get_status_emoji(100-security_score)} |
| 📝 Documentation | {100-doc_score}/100 | {self._get_grade(100-doc_score)} | {self._get_status_emoji(100-doc_score)} |
| ⚡ Performance | {100-perf_score}/100 | {self._get_grade(100-perf_score)} | {self._get_status_emoji(100-perf_score)} |
| 🔗 Integration | {100-int_score}/100 | {self._get_grade(100-int_score)} | {self._get_status_emoji(100-int_score)} |
| **Overall** | **{100-avg_score:.0f}/100** | **{grade}** | **{self._get_status_emoji(100-avg_score)}** |

## 🚨 Key Findings by Agent

"""

        for result in agent_results:
            summary += f"### {result.agent_name} Analysis\n"
            summary += f"- **Severity**: {result.severity_score}/10 {'🔴' if result.severity_score >= 7 else '🟠' if result.severity_score >= 4 else '🟢'}\n"
            summary += f"- **Confidence**: {result.confidence:.0%}\n"
            summary += f"- **Issues Found**: {len(result.findings)}\n"
            summary += f"- **Recommendations**: {len(result.recommendations)}\n\n"

            if result.findings:
                summary += "**Top Issues:**\n"
                for finding in result.findings[:3]:
                    summary += f"- {finding.get('description', 'Issue identified')}\n"
            summary += "\n"

        summary += """## ✅ Quick Actions

- [ ] Review and address security vulnerabilities
- [ ] Complete missing API documentation
- [ ] Implement performance optimizations
- [ ] Standardize error responses
- [ ] Add comprehensive examples

## 📈 Next Steps

1. **Immediate** (This Week):
   - Address critical security issues
   - Fix missing error responses

2. **Short-term** (Next 2 Weeks):
   - Complete API documentation
   - Implement caching strategy

3. **Long-term** (This Month):
   - Optimize performance
   - Add monitoring and metrics
"""
        return summary

    def _get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90: return 'A'
        if score >= 80: return 'B'
        if score >= 70: return 'C'
        if score >= 60: return 'D'
        return 'F'

    def _get_status_emoji(self, score: float) -> str:
        """Get status emoji based on score"""
        if score >= 80: return '✅ Good'
        if score >= 60: return '⚠️ Needs Work'
        return '❌ Critical'

    def _extract_priority_issues(self, agent_results: List[AgentResult]) -> List[Dict[str, Any]]:
        """Extract and prioritize issues across all agents"""
        all_issues = []

        for result in agent_results:
            for finding in result.findings:
                all_issues.append({
                    "issue": finding.get("description", "Unknown issue"),
                    "agent": result.agent_name,
                    "severity": finding.get("severity", "medium"),
                    "category": finding.get("category", result.agent_name),
                    "agent_confidence": result.confidence,
                    "agent_severity_score": result.severity_score
                })

        # Sort by severity and confidence
        severity_order = {"high": 3, "medium": 2, "low": 1}
        all_issues.sort(
            key=lambda x: (
                severity_order.get(x["severity"].lower(), 2),
                x["agent_severity_score"],
                x["agent_confidence"]
            ),
            reverse=True
        )

        return all_issues[:10]  # Top 10 issues

    def _calculate_overall_score(self, agent_results: List[AgentResult]) -> int:
        """Calculate overall API quality score based on agent results"""
        if not agent_results:
            return 0

        # Weight agent scores by their confidence
        weighted_scores = []
        total_weight = 0

        for result in agent_results:
            if result.confidence > 0:
                weight = result.confidence
                # Convert severity to quality score (lower severity = higher quality)
                quality_score = max(0, 100 - (result.severity_score * 10))
                weighted_scores.append(quality_score * weight)
                total_weight += weight

        if total_weight == 0:
            return 50  # Default middle score

        overall_score = sum(weighted_scores) / total_weight
        return int(max(0, min(100, overall_score)))


# Factory function for easy instantiation
def create_agentic_orchestrator(llm_manager: SimpleLLMManager) -> AgenticOrchestrator:
    """Create configured agentic orchestrator"""
    return AgenticOrchestrator(llm_manager)