from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Type


from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from typing import Literal, AsyncIterable, Any

# Bộ nhớ trạng thái
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# --------- Base Input ---------
class MathInput(BaseModel):
    a: float = Field(description="First operand")
    b: float = Field(description="Second operand")


# --------- Tools ---------
class AdditionTool(BaseTool):
    name: str = "addition"
    description: str = "Add two numbers"
    args_schema: Type[BaseModel] = MathInput

    def _run(self, a: float, b: float) -> str:
        return f"{a} + {b} = {a + b}"


class SubtractionTool(BaseTool):
    name: str = "subtraction"
    description: str = "Subtract two numbers"
    args_schema: Type[BaseModel] = MathInput

    def _run(self, a: float, b: float) -> str:
        return f"{a} - {b} = {a - b}"


class MultiplicationTool(BaseTool):
    name: str = "multiplication"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MathInput

    def _run(self, a: float, b: float) -> str:
        return f"{a} * {b} = {a * b}"


class DivisionTool(BaseTool):
    name: str = "division"
    description: str = "Divide two numbers"
    args_schema: Type[BaseModel] = MathInput

    def _run(self, a: float, b: float) -> str:
        if b == 0:
            return "Error: Division by zero"
        return f"{a} / {b} = {a / b}"


# --------- Response Format ---------
class MathResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


# --------- MathAgent sử dụng 4 tool ---------
class MathAgent:
    SYSTEM_INSTRUCTION = (
        "You are a math assistant that solves math problems using the four tools: addition, subtraction, multiplication, division.\n"
        "Decide which tool is appropriate and call it with correct parameters.\n"
        "Only return final result using the message field in the response format.\n"
    )

    RESPONSE_FORMAT_INSTRUCTION = (
        'Respond in the following JSON format:\n'
        '{ "status": "completed", "message": "<insert result here>" }\n'
        'Only use "input_required" if user input is needed. Use "error" if an error occurred.\n'
    )

    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            streaming=True,
        )

        self.tools = [
            AdditionTool(),
            SubtractionTool(),
            MultiplicationTool(),
            DivisionTool()
        ]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, MathResponseFormat),
        )

    async def stream(self, query: str, sessionId: str, client_type: str = 'agent') -> AsyncIterable[dict[str, Any]]:
        '''
            Đang fix cứng là trả dữ liệu về cho agent client
        '''
        inputs: dict[str, Any] = {'messages': [('user', query)]}
        config: RunnableConfig = {
            'configurable': {
                'thread_id': sessionId,
                'client_type': client_type # Đang giao tiếp với agent chứ không phải human
            }
        }

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]

            if client_type != 'agent': 
                if isinstance(message, AIMessage) and message.tool_calls:
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': 'Processing calculation...',
                    }

                elif isinstance(message, ToolMessage):
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': 'Finalizing answer...',
                    }

        yield self.get_agent_response(config)

    def get_agent_response(self, config: RunnableConfig) -> dict[str, Any]:
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')

        if structured_response and isinstance(structured_response, MathResponseFormat):
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }
            elif structured_response.status in {'input_required', 'error'}:
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We could not process your request.',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']