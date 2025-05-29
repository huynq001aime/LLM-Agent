import os

from collections.abc import AsyncIterable
from typing import Any, Literal
from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import (
    RunnableConfig,
)

from langgraph.checkpoint.memory import MemorySaver


memory = MemorySaver()


class SimpleSearchInput(BaseModel):
    query: str = Field(description="should be a search query")
    
    
class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str
    

class SimpleSearchTool(BaseTool):
    name: str = "simple_search"
    description: str = "useful for when you need to answer questions about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        
        # return results
        if not results or "results" not in results:
            return "No results found."

        formatted = ""
        for r in results["results"]:
            formatted += f"- {r.get('title', 'No title')}\n  {r.get('url', '')}\n\n"
        return formatted.strip()


class SimpleSearchAgent:
    SYSTEM_INSTRUCTION = (
    'You are an assistant specializing in searching for current information using the simple_search tool.\n'
    'Your only task is to use the tool to search for user queries and return complete, helpful, and detailed results.\n'
    'When presenting the result, ensure the message field contains the full content extracted from the search results.\n'
    'Avoid vague summaries. Show actual links, headlines, or content if available.\n'
    'Respond using the format defined in the response schema.\n'
    )

    
    RESPONSE_FORMAT_INSTRUCTION: str = (
    'Respond in the following JSON format:\n'
    '{ "status": "completed", "message": "<insert full answer here>" }\n'
    'Include all relevant content from the search results into the "message" field.\n'
    'Do not summarize vaguely; provide detailed and concrete output.'
    'Set status to "input_required" only if user input is required.'
    'Set status to "error" if an error occurs.'
    )
    
    def __init__(self):
        self.model = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0,
                        streaming=True,
                    )
        
        self.tools = [SimpleSearchTool()]
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
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
                '''
                    Chỉ trả về client là human để phục vụ cho UI
                    còn là giao tiếp với agent thì không cần
                '''
                # Nếu LLM gọi tool
                if isinstance(message, AIMessage) and message.tool_calls:
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': 'Searching for information...',
                    }

                # Nếu tool đã trả kết quả và đang xử lý
                elif isinstance(message, ToolMessage):
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': 'Processing search results...',
                    }

        # Sau khi hoàn tất, trả kết quả cuối cùng
        yield self.get_agent_response(config)

    def get_agent_response(self, config: RunnableConfig) -> dict[str, Any]:
        current_state = self.graph.get_state(config)

        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status in {'input_required', 'error'}:
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We are unable to process your request at the moment. Please try again.',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']


