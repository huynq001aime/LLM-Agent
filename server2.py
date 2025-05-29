import os
import sys

import click
import httpx

from agent_math import MathAgent  # type: ignore[import-untyped]
from agent_math_executor import MathAgentExecutor  # type: ignore[import-untyped]
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

load_dotenv()


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10002)
def main(host: str, port: int):
    if not os.getenv('OPENAI_API_KEY'):
        print('OPENAI_API_KEY environment variable not set.')
        sys.exit(1)

    client = httpx.AsyncClient()
    request_handler = DefaultRequestHandler(
        agent_executor=MathAgentExecutor(),
        task_store=InMemoryTaskStore(),
        push_notifier=InMemoryPushNotifier(client),
    )

    server = A2AStarletteApplication(
        agent_card=get_agent_card(host, port),
        http_handler=request_handler,
    )
    import uvicorn

    uvicorn.run(server.build(), host=host, port=port)


def get_agent_card(host: str, port: int) -> AgentCard:
    """Returns the Agent Card for the Math Agent."""
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    skill = AgentSkill(
        id='math_arithmetic',
        name='Perform arithmetic operations',
        description='Performs addition, subtraction, multiplication, and division',
        tags=['arithmetic', 'math', 'calculation', 'multiply', 'divide', 'add', 'subtract'],
        examples=[
            'What is 25 plus 17?',
            'Multiply 9 and 6',
            'Divide 42 by 7',
            'Subtract 88 from 132',
        ],
    )

    return AgentCard(
        name='Math Agent',
        description='Agent for basic arithmetic operations (add, subtract, multiply, divide)',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=MathAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=MathAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )


if __name__ == '__main__':
    main()
