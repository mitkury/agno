# How Agents Work in Agno

This document explains how agents work in Agno, focusing on the agentic tool use and the internals of the agentic loop.

## Overview

Agno agents are autonomous AI entities that can reason, use tools, and interact with users through a sophisticated execution loop. The core agent implementation is found in `libs/agno/agno/agent/agent.py` and provides a comprehensive framework for building intelligent agents.

## Agent Architecture

### Core Components

An Agno agent consists of several key components:

1. **Model**: The underlying language model (OpenAI, Anthropic, etc.)
2. **Tools**: Functions the agent can call to interact with external systems
3. **Memory**: Persistent storage for conversation history and user data
4. **Knowledge**: Vector database for RAG (Retrieval-Augmented Generation)
5. **Reasoning**: Step-by-step problem-solving capabilities
6. **Storage**: Session persistence and state management

### Basic Agent Creation

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    name="Research Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="You are a helpful research assistant.",
    markdown=True,
)
```

## The Agentic Loop

The agentic loop is the core execution mechanism that drives agent behavior. It follows a structured sequence of steps:

### 1. Reasoning Phase (Optional)

If reasoning is enabled (`reasoning=True`), the agent first breaks down the task into steps:

```python
# From libs/agno/agno/agent/agent.py lines 5852-5871
def reason(self, run_messages: RunMessages) -> Iterator[RunResponseEvent]:
    # Yield a reasoning started event
    if self.stream_intermediate_steps:
        yield self._handle_event(
            create_reasoning_started_event(from_run_response=self.run_response), 
            self.run_response
        )
    
    # Get the reasoning model (can be different from main model)
    reasoning_model: Optional[Model] = self.reasoning_model
    if reasoning_model is None and self.model is not None:
        reasoning_model = deepcopy(self.model)
```

The reasoning system uses structured steps defined in `libs/agno/agno/reasoning/step.py`:

```python
class ReasoningStep(BaseModel):
    title: Optional[str] = Field(None, description="A concise title summarizing the step's purpose")
    action: Optional[str] = Field(None, description="The action derived from this step")
    result: Optional[str] = Field(None, description="The result of executing the action")
    reasoning: Optional[str] = Field(None, description="The thought process behind this step")
    next_action: Optional[NextAction] = Field(None, description="Whether to continue, validate, or finalize")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0 to 1.0)")
```

### 2. Model Response Generation

The agent generates a response using the language model, which may include tool calls:

```python
# From libs/agno/agno/agent/agent.py lines 792-801
model_response: ModelResponse = self.model.response(
    messages=run_messages.messages,
    tools=self._tools_for_model,
    functions=self._functions_for_model,
    tool_choice=self.tool_choice,
    tool_call_limit=self.tool_call_limit,
    response_format=response_format,
)
```

### 3. Tool Execution

When the model decides to use tools, the agent executes them through a sophisticated tool handling system:

```python
# From libs/agno/agno/agent/agent.py lines 2558-2573
def _run_tool(self, run_messages: RunMessages, tool: ToolExecution) -> Iterator[RunResponseEvent]:
    # Execute the tool
    function_call = self.model.get_function_call_to_run_from_tool_execution(tool, self._functions_for_model)
    function_call_results: List[Message] = []

    for call_result in self.model.run_function_call(
        function_call=function_call,
        function_call_results=function_call_results,
    ):
        if isinstance(call_result, ModelResponse):
            if call_result.event == ModelResponseEvent.tool_call_started.value:
                yield self._handle_event(
                    create_tool_call_started_event(from_run_response=self.run_response, tool=tool),
                    self.run_response,
                )
```

### 4. Tool Call Management

The agent handles different types of tool interactions:

- **Confirmed Tools**: Tools that require user confirmation before execution
- **External Execution**: Tools that need external systems to execute
- **User Input**: Tools that require user input during execution
- **Paused Tools**: Tools that can pause execution for user interaction

```python
# From libs/agno/agno/agent/agent.py lines 2636-2671
def _handle_tool_call_updates(self, run_response: RunResponse, run_messages: RunMessages):
    for _t in run_response.tools or []:
        # Case 1: Handle confirmed tools and execute them
        if _t.requires_confirmation is not None and _t.requires_confirmation is True:
            if _t.confirmed is not None and _t.confirmed is True and _t.result is None:
                deque(self._run_tool(run_messages, _t), maxlen=0)
            else:
                self._reject_tool_call(run_messages, _t)
        
        # Case 2: Handle external execution required tools
        elif _t.external_execution_required is not None and _t.external_execution_required is True:
            self._handle_external_execution_update(run_messages=run_messages, tool=_t)
        
        # Case 3: Agentic user input required
        elif _t.tool_name == "get_user_input" and _t.requires_user_input is not None:
            self._handle_user_input_update(tool=_t)
            deque(self._run_tool(run_messages, _t), maxlen=0)
```

### 5. Memory and State Management

After tool execution, the agent updates its memory and session state:

```python
# From libs/agno/agno/agent/agent.py lines 774-778
# Steps in the agentic loop:
# 1. Reason about the task if reasoning is enabled
# 2. Generate a response from the Model (includes running function calls)
# 3. Add the run to memory
# 4. Update Agent Memory
# 5. Calculate session metrics
```

## Tool Execution Flow

### Tool Execution Structure

Tools in Agno are represented by the `ToolExecution` class:

```python
# From libs/agno/agno/models/response.py lines 20-48
@dataclass
class ToolExecution:
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_error: Optional[bool] = None
    result: Optional[str] = None
    metrics: Optional[MessageMetrics] = None
    
    # Control flow flags
    stop_after_tool_call: bool = False
    requires_confirmation: Optional[bool] = None
    confirmed: Optional[bool] = None
    requires_user_input: Optional[bool] = None
    external_execution_required: Optional[bool] = None
    
    @property
    def is_paused(self) -> bool:
        return bool(self.requires_confirmation or self.requires_user_input or self.external_execution_required)
```

### Tool Call Events

The system generates events throughout the tool execution lifecycle:

```python
# From libs/agno/agno/models/response.py lines 11-18
class ModelResponseEvent(str, Enum):
    tool_call_paused = "ToolCallPaused"
    tool_call_started = "ToolCallStarted"
    tool_call_completed = "ToolCallCompleted"
    assistant_response = "AssistantResponse"
```

## Agent Invocation

### Basic Usage

Agents can be invoked in several ways:

```python
# Simple response
response = agent.run("What is the weather like?")

# Streaming response
for event in agent.run("Tell me a story", stream=True):
    print(event.content)

# Async usage
response = await agent.arun("Analyze this data")

# With print helper
agent.print_response("Hello world!", stream=True)
```

### Advanced Features

#### Reasoning Agent
```python
reasoning_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True)],
    reasoning=True,  # Enable step-by-step reasoning
    markdown=True,
)

reasoning_agent.print_response(
    "Write a report comparing NVDA to TSLA", 
    stream=True, 
    show_full_reasoning=True
)
```

#### Agent with Memory
```python
memory = Memory(
    model=Claude(id="claude-sonnet-4-20250514"),
    db=SqliteMemoryDb(table_name="user_memories", db_file="tmp/agent.db"),
)

agent = Agent(
    model=Claude(id="claude-sonnet-4-20250514"),
    memory=memory,
    enable_agentic_memory=True,  # Let agent manage its own memories
    user_id="user123",
)
```

## Team Collaboration

Agno supports multi-agent collaboration through teams:

```python
from agno.team.team import Team

web_agent = Agent(
    name="Web Search Agent",
    model=OpenAIChat(id="gpt-4.1"),
    tools=[DuckDuckGoTools()],
    role="Handle web search requests",
)

finance_agent = Agent(
    name="Finance Agent", 
    model=OpenAIChat(id="gpt-4.1"),
    tools=[YFinanceTools(stock_price=True)],
    role="Handle financial data requests",
)

team = Team(
    name="Research Team",
    mode="coordinate",  # or "route", "collaborate"
    model=Claude(id="claude-sonnet-4-20250514"),
    members=[web_agent, finance_agent],
    instructions="Collaborate to provide comprehensive analysis",
)
```

### Team Modes

- **Route**: Route tasks to specific agents based on content
- **Coordinate**: Have agents work together on complex tasks
- **Collaborate**: Agents actively collaborate and build on each other's work

## Workflows

For more complex orchestration, Agno provides workflows:

```python
from agno.workflow import Workflow

class AnalysisWorkflow(Workflow):
    agent = Agent(model=OpenAIChat(id="gpt-4o"))
    
    def run(self, data: str) -> Iterator[RunResponse]:
        # Custom workflow logic
        yield from self.agent.run(f"Analyze: {data}", stream=True)
```

## Key Features

### 1. Streaming Support
Agents support real-time streaming of responses and intermediate steps:

```python
for event in agent.run("Complex task", stream=True, stream_intermediate_steps=True):
    if hasattr(event, 'content'):
        print(event.content)
```

### 2. Tool Confirmation
Tools can require user confirmation before execution:

```python
# Tool execution can be paused for confirmation
if tool.requires_confirmation:
    # Wait for user confirmation
    tool.confirmed = user_confirms()
```

### 3. User Input Integration
Tools can request user input during execution:

```python
if tool.requires_user_input:
    user_input = get_user_input(tool.user_input_schema)
    tool.tool_args.update(user_input)
```

### 4. External Execution
Tools can be executed by external systems:

```python
if tool.external_execution_required:
    # Delegate to external system
    result = external_system.execute(tool)
    tool.result = result
```

## Event System

Agno uses a comprehensive event system for monitoring and debugging:

- `RunResponseStartedEvent`: Agent run begins
- `ReasoningStartedEvent`: Reasoning phase begins
- `ToolCallStartedEvent`: Tool execution begins
- `ToolCallCompletedEvent`: Tool execution completes
- `RunResponseCompletedEvent`: Agent run completes

## Memory Management

Agents maintain several types of memory:

1. **Session Memory**: Conversation history within a session
2. **User Memory**: Persistent user-specific information
3. **Agent Memory**: Agent's own knowledge and experiences
4. **Team Memory**: Shared memory between team members

## Conclusion

Agno's agent system provides a robust framework for building intelligent, tool-using agents. The agentic loop handles reasoning, tool execution, memory management, and user interaction in a structured, event-driven manner. The system supports both simple single-agent scenarios and complex multi-agent collaborations through teams and workflows.

The key to understanding Agno agents is recognizing that they operate through a sophisticated execution loop that can reason about problems, use tools to interact with external systems, manage state and memory, and coordinate with other agents when needed. This makes them capable of handling complex, multi-step tasks that require both intelligence and tool use.