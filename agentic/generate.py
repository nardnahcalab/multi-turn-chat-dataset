#!/usr/bin/env python3
"""
Synthetic multi-turn agentic task dataset generator for benchmarking agent performance.

Generates conversations where agents must complete high-level goals using tool calls,
error recovery, and iterative refinement. Includes traceable success metrics that
punish partial credit and measure end-to-end task completion.

Designed to stress-test:
- Tool-use accuracy and sequencing
- Error handling and recovery
- Goal decomposition and planning
- Iterative refinement and adaptation

Usage:
    python generate.py                     # uses default config.yaml
    python generate.py --config my.yaml    # custom config
    python generate.py --num 1000          # override conversation count
    python generate.py --format parquet    # specific format only
"""

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# ---------------------------------------------------------------------------
# Task-specific conversation templates
# ---------------------------------------------------------------------------

TASK_TEMPLATES = {
    "data_processing": {
        "openers": [
            "I have a CSV file with {num_records} customer records. I need to {goal}. The data has columns: {columns}. Can you help me process this?",
            "I need to transform a dataset where {data_issue}. The goal is to {goal}. What steps would you take?",
            "I have data in {source_format} that needs to be {goal}. The records contain {data_types}. How would you approach this?",
            "Can you help me validate and clean a dataset? The requirements are: {requirements}. The data has {num_records} records.",
            "I need to aggregate data by {grouping} and calculate {metrics}. The source data is in {source_format}. Can you do this?",
            "I have {num_records} records that need to be filtered by {filter_criteria} and then {goal}. How would you do this?",
        ],
        "followups": [
            "I got an error: {error_message}. What went wrong?",
            "The output looks wrong. Can you verify the {field} column?",
            "How many records passed validation? Can you show me the failure details?",
            "Can you also {additional_requirement} while processing?",
            "What if we need to handle {edge_case}? Would the approach change?",
            "Can you export the results to {output_format}?",
            "The transformation is taking too long. Can you optimize it?",
            "I need to apply this transformation to {num_datasets} similar datasets. Can you make it reusable?",
        ],
        "tool_sequences": [
            ["query_database", "validate_schema", "transform_data", "export_data"],
            ["query_database", "transform_data", "validate_schema"],
            ["query_database", "transform_data", "transform_data", "export_data"],
            ["query_database", "validate_schema"],
        ],
        "success_criteria": [
            "All {num_records} records processed without errors",
            "{percentage}% of records passed validation",
            "Data exported to {output_format} successfully",
            "Transformation completed in under {time_limit} seconds",
        ],
        "fill_values": {
            "num_records": ["100", "1000", "10000", "100000"],
            "goal": [
                "remove duplicates and standardize formats",
                "aggregate by customer and calculate totals",
                "filter for records matching specific criteria",
                "join with another dataset and validate consistency",
                "pivot the data and calculate summary statistics",
            ],
            "columns": [
                "id, name, email, created_at, status",
                "product_id, quantity, price, date, region",
                "user_id, action, timestamp, device, country",
            ],
            "data_issue": [
                "there are missing values in critical fields",
                "the date formats are inconsistent",
                "there are duplicate records with slight variations",
                "the numeric fields have mixed types (strings and numbers)",
            ],
            "source_format": ["CSV", "JSON", "Parquet", "SQL database"],
            "data_types": ["strings, integers, dates, and floats", "nested JSON objects", "categorical and numeric fields"],
            "requirements": [
                "remove null values, standardize date formats, validate email addresses",
                "check for duplicates, verify numeric ranges, ensure required fields are present",
                "validate schema, check data types, remove outliers",
            ],
            "grouping": ["customer", "date", "region", "product_type"],
            "metrics": ["sum, count, average", "min, max, percentiles", "distinct count, mode"],
            "filter_criteria": ["status = 'active'", "created_date > 2024-01-01", "amount > 1000"],
            "error_message": [
                "Column 'email' not found in dataset",
                "Type mismatch: expected integer, got string",
                "Query timeout after 30 seconds",
                "Invalid date format in column 'created_at'",
            ],
            "field": ["id", "email", "date", "amount"],
            "additional_requirement": [
                "deduplicate based on email address",
                "add a calculated field for total value",
                "split the name into first and last name",
            ],
            "edge_case": [
                "null values in the grouping column",
                "records with dates in the future",
                "duplicate IDs with different data",
            ],
            "output_format": ["CSV", "JSON", "Parquet", "SQL INSERT statements"],
            "percentage": ["95", "99", "100"],
            "time_limit": ["10", "30", "60"],
            "num_datasets": ["5", "10", "20"],
        },
    },

    "api_integration": {
        "openers": [
            "I need to fetch data from {api_name} API and integrate it into our system. The endpoint is {endpoint}. What's the best approach?",
            "Can you help me call {api_name} API to {goal}? I need to handle {challenge}.",
            "I'm trying to integrate with {api_name} but I'm getting {error}. The API documentation says {api_detail}.",
            "I need to fetch data from {api_name}, transform it, and load it into {destination}. How would you structure this?",
            "Can you help me build a reliable integration with {api_name}? We need to handle {challenge}.",
        ],
        "followups": [
            "I got a {error_code} error. What does that mean?",
            "The API is rate-limited to {rate_limit}. How should we handle this?",
            "The response format is {response_format}. Can you parse it correctly?",
            "Some requests are timing out. Should we add retry logic?",
            "The API returns {num_records} records but we only need {subset}. Can you filter it?",
            "Can we cache the results to avoid repeated calls?",
            "The API requires authentication. How should we handle credentials?",
        ],
        "tool_sequences": [
            ["call_api", "parse_response", "validate_schema", "export_data"],
            ["call_api", "parse_response", "transform_data", "export_data"],
            ["call_api", "retry_with_backoff", "parse_response", "export_data"],
            ["call_api", "parse_response"],
        ],
        "success_criteria": [
            "Successfully fetched {num_records} records from API",
            "All {num_records} records parsed and validated",
            "Data integrated into {destination} without errors",
            "Integration completed with {num_retries} retries",
        ],
        "fill_values": {
            "api_name": ["GitHub", "Twitter", "OpenWeather", "CoinGecko", "NewsAPI"],
            "endpoint": ["/users/{username}", "/repos/{owner}/{repo}/issues", "/data/weather", "/simple/price"],
            "goal": [
                "get all issues for a repository",
                "fetch user profile information",
                "get current weather data",
                "retrieve cryptocurrency prices",
            ],
            "challenge": [
                "rate limiting",
                "authentication",
                "pagination",
                "error handling",
                "data transformation",
            ],
            "error": [
                "401 Unauthorized",
                "429 Too Many Requests",
                "500 Internal Server Error",
                "Invalid response format",
            ],
            "api_detail": [
                "requires Bearer token authentication",
                "returns paginated results with max 100 per page",
                "rate limited to 60 requests per minute",
                "response is nested JSON with optional fields",
            ],
            "destination": ["PostgreSQL database", "Elasticsearch", "data warehouse", "cache layer"],
            "error_code": ["401", "429", "500", "503"],
            "rate_limit": ["100 requests/minute", "1000 requests/hour", "10 requests/second"],
            "response_format": ["JSON", "XML", "CSV", "nested JSON with arrays"],
            "num_records": ["100", "1000", "10000"],
            "subset": ["active records", "records from last 7 days", "records matching criteria"],
            "num_retries": ["0", "1", "2", "3"],
        },
    },

    "system_troubleshooting": {
        "openers": [
            "Our {system_component} is having issues. The symptoms are: {symptoms}. Can you help diagnose?",
            "I'm seeing errors in the logs: {error_pattern}. What could be causing this?",
            "The {system_component} performance has degraded. {metric} is at {value}. What should we check?",
            "We're getting {error_type} errors. The logs show {log_snippet}. How do we fix this?",
            "Can you troubleshoot why {system_component} is failing? The error message is {error_message}.",
        ],
        "followups": [
            "I checked the logs and found {finding}. What does this mean?",
            "The issue started after we {recent_change}. Could that be related?",
            "I tried {attempted_fix} but it didn't work. What else should we try?",
            "How critical is this issue? What's the impact?",
            "Can you provide a permanent fix or just a workaround?",
            "Should we roll back the recent changes?",
        ],
        "tool_sequences": [
            ["check_logs", "diagnose_issue", "apply_fix", "verify_resolution"],
            ["check_logs", "diagnose_issue"],
            ["diagnose_issue", "apply_fix", "verify_resolution"],
            ["check_logs", "apply_fix"],
        ],
        "success_criteria": [
            "Issue diagnosed and root cause identified",
            "Fix applied and verified to resolve the issue",
            "{metric} returned to normal levels",
            "System operational with no errors in logs",
        ],
        "fill_values": {
            "system_component": ["database", "API server", "cache layer", "message queue", "load balancer"],
            "symptoms": [
                "slow response times, high CPU usage",
                "connection timeouts, connection pool exhausted",
                "memory usage increasing over time",
                "intermittent failures, errors in logs",
            ],
            "error_pattern": [
                "OutOfMemoryError in application logs",
                "Connection refused errors",
                "Timeout waiting for database connection",
                "Disk space full errors",
            ],
            "metric": ["CPU usage", "memory usage", "response time", "error rate"],
            "value": ["95%", "8GB/10GB", "2000ms", "5%"],
            "error_type": ["timeout", "connection refused", "out of memory", "disk full"],
            "log_snippet": [
                "FATAL: out of memory",
                "ERROR: failed to connect to database",
                "WARNING: connection pool exhausted",
            ],
            "error_message": [
                "Connection timeout after 30 seconds",
                "Disk space exhausted",
                "Out of memory: Java heap space",
            ],
            "finding": [
                "multiple failed connection attempts",
                "memory usage growing linearly",
                "spike in error rate at specific time",
            ],
            "recent_change": [
                "deployed a new version",
                "increased traffic",
                "added a new feature",
                "changed database configuration",
            ],
            "attempted_fix": [
                "restarted the service",
                "cleared the cache",
                "increased memory allocation",
            ],
        },
    },

    "code_generation": {
        "openers": [
            "I need to write a {language} function that {requirement}. The function should {details}.",
            "Can you help me write {language} code to {goal}? It needs to {constraint}.",
            "I need a {language} script that {task}. The input will be {input_type} and output should be {output_type}.",
            "Can you generate {language} code for {feature}? It should handle {edge_cases}.",
        ],
        "followups": [
            "Can you add error handling for {error_case}?",
            "The code needs to be optimized for {optimization_goal}. Can you refactor it?",
            "Can you add unit tests for this code?",
            "I ran the code but got {error}. What's wrong?",
            "Can you add logging and debugging output?",
            "The code should also handle {additional_requirement}.",
            "Can you make this function reusable for {use_case}?",
        ],
        "tool_sequences": [
            ["write_code", "run_tests", "debug_code"],
            ["write_code", "execute_code"],
            ["write_code", "run_tests"],
            ["write_code", "execute_code", "debug_code"],
        ],
        "success_criteria": [
            "Code generated and syntax is valid",
            "{num_tests} unit tests pass",
            "Code executes without errors",
            "Code handles all specified edge cases",
        ],
        "fill_values": {
            "language": ["Python", "JavaScript", "Go", "Rust", "Java"],
            "requirement": [
                "calculates the factorial of a number",
                "parses JSON and extracts specific fields",
                "connects to a database and runs a query",
                "implements a binary search algorithm",
            ],
            "details": [
                "handle edge cases like negative numbers",
                "validate input and raise appropriate errors",
                "be efficient for large inputs",
                "include comprehensive documentation",
            ],
            "goal": [
                "process a CSV file",
                "make HTTP requests",
                "manipulate strings and arrays",
                "interact with a database",
            ],
            "constraint": [
                "be memory efficient",
                "complete in under 1 second",
                "handle concurrent requests",
                "work with large datasets",
            ],
            "task": [
                "reads a file and counts word frequencies",
                "fetches data from an API and processes it",
                "generates a report from structured data",
            ],
            "input_type": ["JSON", "CSV", "plain text", "binary data"],
            "output_type": ["JSON", "CSV", "formatted text", "HTML"],
            "feature": [
                "user authentication",
                "data validation",
                "caching mechanism",
                "rate limiting",
            ],
            "edge_cases": [
                "empty inputs",
                "null values",
                "very large inputs",
                "invalid data types",
            ],
            "error_case": [
                "null pointer exceptions",
                "division by zero",
                "file not found",
                "network timeout",
            ],
            "optimization_goal": [
                "memory usage",
                "execution speed",
                "readability",
            ],
            "error": [
                "TypeError: unsupported operand type",
                "IndexError: list index out of range",
                "NameError: name is not defined",
            ],
            "additional_requirement": [
                "support multiple input formats",
                "add configuration options",
                "implement retry logic",
            ],
            "use_case": [
                "different data sources",
                "various output formats",
                "different programming languages",
            ],
            "num_tests": ["3", "5", "10"],
        },
    },

    "research_synthesis": {
        "openers": [
            "I need to research {topic} and create a comprehensive report. The report should cover {aspects}.",
            "Can you help me gather information about {topic}? I need to understand {goal}.",
            "I'm writing about {topic} and need to synthesize information from multiple sources. Key areas: {areas}.",
            "Can you research {topic} and provide a summary of {key_points}?",
        ],
        "followups": [
            "Can you find more information about {specific_aspect}?",
            "I need sources for the claims about {claim}. Can you find references?",
            "Can you compare {topic_a} and {topic_b}?",
            "Can you explain {concept} in simpler terms?",
            "I need more recent information. Can you search for {topic} from the last {time_period}?",
            "Can you organize the findings by {organization_method}?",
        ],
        "tool_sequences": [
            ["search_knowledge_base", "fetch_document", "summarize_content", "generate_report"],
            ["search_knowledge_base", "fetch_document", "summarize_content"],
            ["search_knowledge_base", "generate_report"],
            ["search_knowledge_base", "fetch_document"],
        ],
        "success_criteria": [
            "Found {num_sources} relevant sources",
            "Report covers all {num_aspects} required aspects",
            "Synthesis includes {num_perspectives} different perspectives",
            "Report generated with proper citations",
        ],
        "fill_values": {
            "topic": ["machine learning", "climate change", "artificial intelligence", "blockchain", "quantum computing"],
            "aspects": [
                "history, current state, future trends",
                "technical details, applications, limitations",
                "pros and cons, ethical considerations",
            ],
            "goal": [
                "the key concepts and how they work",
                "the current state of the art",
                "the practical applications",
            ],
            "areas": [
                "technical foundations, applications, challenges",
                "history, current research, future directions",
                "benefits, risks, ethical implications",
            ],
            "key_points": [
                "main findings, trends, future outlook",
                "advantages and disadvantages",
                "key players and innovations",
            ],
            "specific_aspect": [
                "implementation details",
                "real-world examples",
                "recent developments",
            ],
            "claim": [
                "that this technology is revolutionary",
                "that this approach is more efficient",
                "that this has significant limitations",
            ],
            "topic_a": ["approach A", "technology X", "method 1"],
            "topic_b": ["approach B", "technology Y", "method 2"],
            "concept": ["neural networks", "consensus mechanisms", "quantum entanglement"],
            "time_period": ["6 months", "1 year", "2 years"],
            "organization_method": [
                "chronological order",
                "by importance",
                "by category",
            ],
            "num_sources": ["5", "10", "15"],
            "num_aspects": ["3", "5", "7"],
            "num_perspectives": ["2", "3", "4"],
        },
    },

    "planning_execution": {
        "openers": [
            "I have a high-level goal: {goal}. Can you break this down into actionable steps?",
            "I need to accomplish {objective} with constraints: {constraints}. How would you plan this?",
            "Can you help me plan {project_type}? The requirements are {requirements}.",
            "I need to execute {task} efficiently. What's the best plan?",
        ],
        "followups": [
            "I've completed {completed_step}. What's next?",
            "I ran into {obstacle}. How should we adapt the plan?",
            "Can we parallelize {steps}?",
            "What's the critical path for this plan?",
            "How long will each step take?",
            "What could go wrong at {step}?",
            "Can we skip {step} or combine it with {other_step}?",
        ],
        "tool_sequences": [
            ["decompose_task", "execute_step", "track_progress", "adapt_plan"],
            ["decompose_task", "execute_step", "track_progress"],
            ["decompose_task", "execute_step"],
            ["decompose_task", "adapt_plan"],
        ],
        "success_criteria": [
            "Objective achieved: {objective}",
            "All {num_steps} steps completed successfully",
            "Project completed {time_comparison} than estimated",
            "No critical issues encountered",
        ],
        "fill_values": {
            "goal": [
                "launch a new product feature",
                "migrate to a new technology stack",
                "implement a data pipeline",
                "build a new system",
            ],
            "objective": [
                "complete the project",
                "achieve the target metrics",
                "deliver on time and budget",
            ],
            "constraints": [
                "limited budget and timeline",
                "team of 5 people",
                "must maintain backward compatibility",
                "zero downtime required",
            ],
            "project_type": [
                "software migration",
                "infrastructure upgrade",
                "feature development",
                "system redesign",
            ],
            "requirements": [
                "must be completed in 4 weeks",
                "must support 1M concurrent users",
                "must maintain 99.9% uptime",
            ],
            "task": [
                "a complex data migration",
                "a system redesign",
                "a product launch",
            ],
            "completed_step": [
                "the planning phase",
                "the design review",
                "the initial implementation",
            ],
            "obstacle": [
                "unexpected technical challenge",
                "resource constraint",
                "dependency delay",
                "requirement change",
            ],
            "steps": [
                "data migration and validation",
                "testing and deployment",
                "documentation and training",
            ],
            "step": [
                "the critical database migration",
                "the user acceptance testing",
                "the production deployment",
            ],
            "other_step": [
                "the documentation phase",
                "the training phase",
            ],
            "time_comparison": [
                "ahead of schedule",
                "on schedule",
                "slightly behind schedule",
            ],
            "num_steps": ["5", "8", "12"],
        },
    },
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def fill_template(template: str, fill_values: Dict[str, List[str]], rng: random.Random) -> str:
    """Fill a template with random values from fill_values dict."""
    result = template
    for key, values in fill_values.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, rng.choice(values))
    return result


def generate_tool_call(
    tool_name: str, task_type: str, rng: random.Random
) -> Dict[str, Any]:
    """Generate a realistic tool call with parameters."""
    tool_calls = {
        "query_database": {
            "tool": "query_database",
            "parameters": {
                "query": rng.choice([
                    "SELECT * FROM users WHERE status = 'active'",
                    "SELECT COUNT(*) FROM orders WHERE date > '2024-01-01'",
                    "SELECT product_id, SUM(quantity) FROM sales GROUP BY product_id",
                ]),
                "timeout": rng.choice([30, 60, 120]),
            },
        },
        "transform_data": {
            "tool": "transform_data",
            "parameters": {
                "operation": rng.choice(["filter", "map", "aggregate", "join"]),
                "config": {
                    "field": rng.choice(["status", "date", "amount", "category"]),
                    "value": rng.choice(["active", "2024-01-01", "1000", "electronics"]),
                },
            },
        },
        "validate_schema": {
            "tool": "validate_schema",
            "parameters": {
                "strict": rng.choice([True, False]),
                "schema": {
                    "required_fields": rng.choice([
                        ["id", "name", "email"],
                        ["product_id", "quantity", "price"],
                    ]),
                },
            },
        },
        "export_data": {
            "tool": "export_data",
            "parameters": {
                "format": rng.choice(["csv", "json", "parquet"]),
                "destination": rng.choice(["/tmp/output.csv", "s3://bucket/data.json"]),
            },
        },
        "call_api": {
            "tool": "call_api",
            "parameters": {
                "url": rng.choice([
                    "https://api.github.com/repos/owner/repo/issues",
                    "https://api.weather.com/forecast",
                    "https://api.example.com/data",
                ]),
                "method": rng.choice(["GET", "POST"]),
                "timeout": rng.choice([30, 60]),
            },
        },
        "parse_response": {
            "tool": "parse_response",
            "parameters": {
                "format": rng.choice(["json", "xml", "html"]),
            },
        },
        "retry_with_backoff": {
            "tool": "retry_with_backoff",
            "parameters": {
                "max_retries": rng.choice([3, 5, 10]),
                "initial_delay": rng.choice([1.0, 2.0]),
            },
        },
        "check_logs": {
            "tool": "check_logs",
            "parameters": {
                "log_type": rng.choice(["system", "application", "error"]),
                "pattern": rng.choice(["ERROR", "TIMEOUT", "FAILED"]),
                "limit": rng.choice([50, 100, 200]),
            },
        },
        "diagnose_issue": {
            "tool": "diagnose_issue",
            "parameters": {
                "system_component": rng.choice(["cpu", "memory", "disk", "network"]),
            },
        },
        "apply_fix": {
            "tool": "apply_fix",
            "parameters": {
                "fix_type": rng.choice(["restart", "reconfigure", "patch"]),
            },
        },
        "verify_resolution": {
            "tool": "verify_resolution",
            "parameters": {
                "verification_method": rng.choice(["test", "check", "monitor"]),
                "duration": rng.choice([30, 60, 120]),
            },
        },
        "write_code": {
            "tool": "write_code",
            "parameters": {
                "language": rng.choice(["python", "javascript", "go"]),
            },
        },
        "execute_code": {
            "tool": "execute_code",
            "parameters": {
                "language": rng.choice(["python", "javascript", "go"]),
                "timeout": rng.choice([30, 60]),
            },
        },
        "run_tests": {
            "tool": "run_tests",
            "parameters": {
                "test_framework": rng.choice(["pytest", "jest", "go test"]),
            },
        },
        "search_knowledge_base": {
            "tool": "search_knowledge_base",
            "parameters": {
                "knowledge_base": rng.choice(["documentation", "wiki", "research"]),
                "limit": rng.choice([5, 10, 20]),
            },
        },
        "fetch_document": {
            "tool": "fetch_document",
            "parameters": {
                "format": rng.choice(["text", "html", "pdf"]),
            },
        },
        "generate_report": {
            "tool": "generate_report",
            "parameters": {
                "format": rng.choice(["markdown", "html", "pdf"]),
            },
        },
        "decompose_task": {
            "tool": "decompose_task",
            "parameters": {},
        },
        "execute_step": {
            "tool": "execute_step",
            "parameters": {},
        },
        "track_progress": {
            "tool": "track_progress",
            "parameters": {},
        },
        "adapt_plan": {
            "tool": "adapt_plan",
            "parameters": {},
        },
    }
    return tool_calls.get(tool_name, {"tool": tool_name, "parameters": {}})


def generate_tool_result(
    tool_name: str, success: bool, rng: random.Random
) -> Dict[str, Any]:
    """Generate a realistic tool result (success or failure)."""
    if success:
        results = {
            "query_database": {
                "status": "success",
                "rows_returned": rng.randint(10, 1000),
                "execution_time_ms": rng.randint(100, 5000),
            },
            "transform_data": {
                "status": "success",
                "records_processed": rng.randint(100, 10000),
                "records_transformed": rng.randint(50, 9000),
            },
            "validate_schema": {
                "status": "success",
                "records_valid": rng.randint(900, 1000),
                "records_invalid": rng.randint(0, 100),
                "validation_errors": [],
            },
            "export_data": {
                "status": "success",
                "file_path": "/tmp/output.csv",
                "file_size_bytes": rng.randint(1000, 1000000),
            },
            "call_api": {
                "status": "success",
                "http_status": 200,
                "response_time_ms": rng.randint(100, 2000),
                "records_fetched": rng.randint(10, 1000),
            },
            "parse_response": {
                "status": "success",
                "records_parsed": rng.randint(10, 1000),
                "parsing_errors": 0,
            },
            "run_tests": {
                "status": "success",
                "total_tests": rng.randint(5, 50),
                "passed": rng.randint(4, 50),
                "failed": 0,
            },
        }
        return results.get(tool_name, {"status": "success"})
    else:
        errors = {
            "query_database": {
                "status": "error",
                "error_type": rng.choice(["timeout", "connection_error", "syntax_error"]),
                "error_message": "Query execution failed",
            },
            "call_api": {
                "status": "error",
                "error_type": rng.choice(["timeout", "connection_error", "rate_limit"]),
                "http_status": rng.choice([408, 429, 500, 503]),
                "error_message": "API request failed",
            },
            "validate_schema": {
                "status": "error",
                "records_valid": rng.randint(500, 900),
                "records_invalid": rng.randint(100, 500),
                "validation_errors": [
                    "Missing required field: email",
                    "Type mismatch in field: age",
                ],
            },
            "run_tests": {
                "status": "error",
                "total_tests": rng.randint(5, 50),
                "passed": rng.randint(0, 40),
                "failed": rng.randint(1, 20),
            },
        }
        return errors.get(tool_name, {"status": "error", "error_message": "Operation failed"})


def calculate_success_score(
    task_type: str, num_turns: int, tool_calls: List[Dict], rng: random.Random
) -> Tuple[float, str]:
    """Calculate success score based on task type and execution."""
    # Base score depends on task completion
    base_score = 0.7 + rng.random() * 0.3  # 0.7-1.0

    # Penalize for errors in tool calls
    error_count = sum(1 for call in tool_calls if call.get("result", {}).get("status") == "error")
    if error_count > 0:
        base_score *= (1 - 0.1 * error_count)

    # Penalize for incomplete tasks (fewer turns than expected)
    if num_turns < 3:
        base_score *= 0.8

    # Clamp to [0, 1]
    final_score = max(0.0, min(1.0, base_score))

    # Determine metric based on task type
    metrics = {
        "data_processing": "data_integrity_score",
        "api_integration": "integration_completeness_score",
        "system_troubleshooting": "resolution_success_score",
        "code_generation": "test_pass_rate",
        "research_synthesis": "coverage_completeness_score",
        "planning_execution": "objective_completion_score",
    }

    return final_score, metrics.get(task_type, "completion_score")


def generate_conversation(
    task_type: str, num_turns: int, rng: random.Random, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a single agentic task conversation."""
    task_config = TASK_TEMPLATES[task_type]
    fill_values = task_config["fill_values"]

    # Generate initial task description (user message)
    opener = rng.choice(task_config["openers"])
    initial_task = fill_template(opener, fill_values, rng)

    # Select tool sequence for this conversation
    tool_sequence = rng.choice(task_config["tool_sequences"])

    # Build messages array
    messages = [
        {
            "role": "system",
            "content": config["task_types"][
                next(i for i, t in enumerate(config["task_types"]) if t["name"] == task_type)
            ]["system_prompt"],
        },
        {"role": "user", "content": initial_task},
    ]

    # Generate turns
    tool_calls_log = []
    for turn_idx in range(num_turns - 1):
        # Agent response with tool calls
        num_tool_calls = rng.choices(
            [0, 1, 2, 3],
            weights=[
                config["tool_calls"]["distribution"]["zero_calls"],
                config["tool_calls"]["distribution"]["one_call"],
                config["tool_calls"]["distribution"]["two_calls"],
                config["tool_calls"]["distribution"]["three_plus"],
            ],
        )[0]

        agent_response = "I'll help you with this task. Let me break it down:\n\n"

        current_tool_calls = []
        for _ in range(num_tool_calls):
            tool_name = rng.choice(tool_sequence)
            tool_call = generate_tool_call(tool_name, task_type, rng)

            # Determine if tool call succeeds (with error injection)
            success = rng.random() > config["error_injection"]["failure_rate"]
            result = generate_tool_result(tool_name, success, rng)

            current_tool_calls.append({
                "tool": tool_name,
                "call": tool_call,
                "result": result,
            })

            # Add to response
            status_str = "✓" if success else "✗"
            agent_response += f"{status_str} {tool_name}: {result.get('status', 'unknown')}\n"

        tool_calls_log.extend(current_tool_calls)
        agent_response += "\nBased on these results, here's my analysis and next steps..."

        messages.append({"role": "assistant", "content": agent_response})

        # User follow-up (if not the last turn)
        if turn_idx < num_turns - 2:
            followup = rng.choice(task_config["followups"])
            followup_text = fill_template(followup, fill_values, rng)
            messages.append({"role": "user", "content": followup_text})

    # Calculate success metrics
    success_score, metric_name = calculate_success_score(task_type, num_turns, tool_calls_log, rng)

    # Generate cumulative character lengths for context growth tracking
    cumulative_lengths = []
    total_chars = 0
    for msg in messages:
        total_chars += len(msg["content"])
        cumulative_lengths.append(total_chars)

    return {
        "conversation_id": str(uuid.uuid4()),
        "task_type": task_type,
        "num_turns": num_turns,
        "num_messages": len(messages),
        "system_prompt": messages[0]["content"],
        "messages": json.dumps(messages),
        "tool_calls": json.dumps(tool_calls_log),
        "total_characters": total_chars,
        "estimated_tokens": total_chars // 4,
        "cumulative_char_lengths": json.dumps(cumulative_lengths),
        "success_metric": metric_name,
        "success_score": success_score,
        "num_tool_calls": len(tool_calls_log),
        "num_errors": sum(1 for call in tool_calls_log if call["result"].get("status") == "error"),
    }


def generate_dataset(
    num_conversations: int, config: Dict[str, Any], seed: int = 42
) -> pd.DataFrame:
    """Generate the complete agentic task dataset."""
    rng = random.Random(seed)

    # Calculate distribution of conversations across task types
    task_types = config["task_types"]
    task_weights = [t["weight"] for t in task_types]
    task_names = [t["name"] for t in task_types]

    # Calculate distribution of conversations across turn counts
    turn_distribution = config["turns"]["distribution"]
    turn_buckets = [
        (turn_distribution["short"]["min_turns"], turn_distribution["short"]["max_turns"],
         turn_distribution["short"]["count"]),
        (turn_distribution["medium"]["min_turns"], turn_distribution["medium"]["max_turns"],
         turn_distribution["medium"]["count"]),
        (turn_distribution["long"]["min_turns"], turn_distribution["long"]["max_turns"],
         turn_distribution["long"]["count"]),
    ]

    conversations = []
    for _ in range(num_conversations):
        # Select task type based on weights
        task_type = rng.choices(task_names, weights=task_weights)[0]

        # Select turn count based on distribution
        bucket = rng.choice(turn_buckets)
        num_turns = rng.randint(bucket[0], bucket[1])

        # Generate conversation
        conversation = generate_conversation(task_type, num_turns, rng, config)
        conversations.append(conversation)

    return pd.DataFrame(conversations)


def save_dataset(
    df: pd.DataFrame, output_dir: str, output_filename: str, formats: List[str] = None
) -> None:
    """Save dataset in multiple formats."""
    if formats is None:
        formats = ["parquet", "aiperf", "mooncake"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parquet format
    if "parquet" in formats:
        base_name = output_filename.replace(".parquet", "")
        parquet_file = output_path / (base_name + ".parquet")
        df.to_parquet(parquet_file, index=False)
        print(f"✓ Saved Parquet: {parquet_file}")

    # aiperf multi_turn format (JSONL)
    if "aiperf" in formats:
        base_name = output_filename.replace(".parquet", "")
        aiperf_file = output_path / (base_name + ".jsonl")
        with open(aiperf_file, "w") as f:
            for _, row in df.iterrows():
                messages = json.loads(row["messages"])
                # Extract user messages for multi_turn format
                user_messages = [
                    {"text": msg["content"]}
                    for msg in messages
                    if msg["role"] == "user"
                ]
                aiperf_record = {
                    "session_id": row["conversation_id"],
                    "turns": user_messages,
                }
                f.write(json.dumps(aiperf_record) + "\n")
        print(f"✓ Saved aiperf JSONL: {aiperf_file}")

    # mooncake_trace format (JSONL with full message arrays)
    if "mooncake" in formats:
        base_name = output_filename.replace(".parquet", "")
        mooncake_file = output_path / (base_name + "_mooncake.jsonl")
        with open(mooncake_file, "w") as f:
            for _, row in df.iterrows():
                messages = json.loads(row["messages"])
                # Output one line per turn (after user message)
                for i in range(1, len(messages), 2):
                    if i + 1 < len(messages):
                        mooncake_record = {
                            "session_id": row["conversation_id"],
                            "messages": messages[: i + 2],
                            "output_length": len(messages[i + 1]["content"].split()),
                        }
                        f.write(json.dumps(mooncake_record) + "\n")
        print(f"✓ Saved mooncake JSONL: {mooncake_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic agentic task dataset for agent benchmarking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Override number of conversations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "aiperf", "mooncake", "all"],
        default="all",
        help="Output format(s)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.num:
        config["dataset"]["num_conversations"] = args.num
    if args.seed:
        config["dataset"]["seed"] = args.seed

    num_conversations = config["dataset"]["num_conversations"]
    seed = config["dataset"]["seed"]
    output_dir = Path(__file__).parent / config["dataset"]["output_dir"]
    output_filename = config["dataset"]["output_filename"]

    print(f"Generating {num_conversations} agentic task conversations (seed={seed})...")
    df = generate_dataset(num_conversations, config, seed=seed)

    print(f"Dataset shape: {df.shape}")
    print(f"Task type distribution:\n{df['task_type'].value_counts()}")
    print(f"Turn count statistics:\n{df['num_turns'].describe()}")
    print(f"Success score statistics:\n{df['success_score'].describe()}")

    # Determine formats to save
    formats = ["parquet", "aiperf", "mooncake"] if args.format == "all" else [args.format]

    print(f"\nSaving dataset in formats: {formats}")
    save_dataset(df, str(output_dir), output_filename, formats)

    print("\n✓ Dataset generation complete!")


if __name__ == "__main__":
    main()
