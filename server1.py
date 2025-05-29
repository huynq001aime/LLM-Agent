import os
import sys

import click
import httpx

from agent_search import SimpleSearchAgent  # type: ignore[import-untyped]
from agent_search_executor import SimpleSearchAgentExecutor  # type: ignore[import-untyped]
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

load_dotenv()

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host: str, port: int):
    if not os.getenv('OPENAI_API_KEY'):
        print('OPENAI_API_KEY environment variable not set.')
        sys.exit(1)

    client = httpx.AsyncClient()
    request_handler = DefaultRequestHandler(
        agent_executor=SimpleSearchAgentExecutor(),
        task_store=InMemoryTaskStore(),
        push_notifier=InMemoryPushNotifier(client),
    )

    server = A2AStarletteApplication(
        agent_card=get_agent_card(host, port), http_handler=request_handler
    )
    import uvicorn

    uvicorn.run(server.build(), host=host, port=port)


def get_agent_card(host: str, port: int):
    """Returns the Agent Card for the Currency Agent."""
    capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
    skill = AgentSkill(
        id='search_information',
        name='Search the latest information',
        description='Helps search the latest information about a particular subject',
        tags=['search for subject', 'information about the subject'],
        examples=['Search the latest information about Donald Trump?', 'Search Manchester City transfer news?'],
    )
    return AgentCard(
        name='Simple Search Agent',
        description='Helps search the latest information about a particular subject',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=SimpleSearchAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=SimpleSearchAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )


if __name__ == '__main__':
    main()
