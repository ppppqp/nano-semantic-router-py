from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: Any
    strict: Optional[bool] = None


@dataclass
class Tool:
    type: str
    function: str
    function_definition: FunctionDefinition


@dataclass
class ResponseAPIRequest:
    model: str
    input: str
    previous_response_id: Optional[str] = None
    instructions: Optional[str] = None
    store: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None
    metadata: Optional[Any] = None
    stream: Optional[bool] = None
    conversation_id: Optional[str] = None


@dataclass
class ContentPart:
    type: str
    text: Optional[str] = None
    annotations: Optional[List[Any]] = None


@dataclass
class OutputItem:
    type: str
    id: str
    role: str
    content: Optional[List[ContentPart]] = None
    status: Optional[str] = None
    name: Optional[str] = None
    call_id: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None


@dataclass
class ResponseAPIResponse:
    id: str
    object: str
    created_at: int
    model: str
    status: str
    output: Optional[List[OutputItem]] = None
    tool_calls: Optional[List[Any]] = None
    metadata: Optional[Any] = None
