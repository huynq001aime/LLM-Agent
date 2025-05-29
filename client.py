import asyncio
from a2a.client import A2AClient
from a2a.types import MessageSendParams, SendStreamingMessageRequest, AgentCard
from typing import List, Dict, Literal, Union, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel
from uuid import uuid4
import httpx
import json

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

AGENT_ENDPOINTS = [
    "http://localhost:10000",
    "http://localhost:10002",
]

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def extract_text_from_chunk(chunk: dict) -> str:
    texts = []
    result = chunk.get("result", {})

    for section in ["history", "status", "artifact"]:
        part_container = result.get(section, {})
        if section == "status":
            part_container = part_container.get("message", {})

        if isinstance(part_container, list):
            parts = part_container
        elif isinstance(part_container, dict):
            parts = part_container.get("parts", [])
        else:
            parts = []

        for part in parts:
            if isinstance(part, dict) and part.get("kind") == "text":
                texts.append(part.get("text", ""))

    return " ".join(texts).strip()

async def load_all_agents(httpx_client: httpx.AsyncClient, endpoints: List[str]) -> List[Dict]:
    agents_info = []
    for endpoint in endpoints:
        url = f"{endpoint}/.well-known/agent.json"
        response = await httpx_client.get(url)
        agent_card = AgentCard(**response.json())

        all_tags = []
        for skill in agent_card.skills or []:
            if skill.tags:
                all_tags.extend(skill.tags)
        unique_tags = list(set(all_tags))

        client = await A2AClient.get_client_from_agent_card_url(httpx_client, endpoint)
        agents_info.append({
            "name": agent_card.name,
            "description": agent_card.description,
            "tags": unique_tags,
            "client": client,
        })
    return agents_info

def make_supervisor_node(agents_info: List[Dict]):
    agent_desc = "\n".join(
        [f'- "{a["name"]}": {a["description"]} (tags: {", ".join(a["tags"])})' for a in agents_info]
    )
    system_prompt = f"""
        You are a supervisor agent. Given a user's question, select the most appropriate agent to handle it.
        Choose from the following:
        {agent_desc}
        Return only the agent name as a single word, or \"END\" to stop.
    """

    AgentNames = Literal[tuple(a["name"] for a in agents_info)]
    class Router(BaseModel):
        next: Union[AgentNames, Literal["END"]]  # type: ignore

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    router = llm.with_structured_output(Router)

    async def supervisor(state: dict) -> dict:
        messages: List[BaseMessage] = state.get("messages", [])
        user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        formatted_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ]
        result = await router.ainvoke(formatted_messages)
        return {"next": result.next}

    return supervisor

def make_agent_node(agent_name: str, client: A2AClient):
    async def node(state: dict) -> dict:
        messages: List[BaseMessage] = state.get("messages", [])
        user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": user_msg}],
                "messageId": uuid4().hex,
            }
        }
        request = SendStreamingMessageRequest(params=MessageSendParams(**payload))
        async for raw_chunk in client.send_message_streaming(request):
            if hasattr(raw_chunk, 'root'):
                chunk_dict = raw_chunk.root.model_dump(mode="python", exclude_none=True)
            else:
                chunk_dict = raw_chunk.model_dump(mode="python", exclude_none=True)
            print(chunk_dict)
            text = extract_text_from_chunk(chunk_dict)
            if text:
                return {"messages": [AIMessage(content=text)]}
        return {}
    return node

def build_workflow(agents_info: List[Dict]):
    workflow = StateGraph(state_schema=State)
    supervisor = make_supervisor_node(agents_info)
    workflow.add_node("supervisor", supervisor)
    workflow.set_entry_point("supervisor")

    for agent in agents_info:
        node_fn = make_agent_node(agent["name"], agent["client"])
        workflow.add_node(agent["name"], node_fn)
        workflow.add_edge(agent["name"], END)

    transitions = {a["name"]: a["name"] for a in agents_info}
    transitions["END"] = END
    workflow.add_conditional_edges("supervisor", lambda output: output.get("next", "END"), transitions)
    return workflow

async def run():
    consumer = AIOKafkaConsumer(
        'qachat',
        bootstrap_servers='localhost:9092',
        group_id='agent2',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=False,
    )
    await consumer.start()

    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        linger_ms=0,
    )
    await producer.start()
    
    async with httpx.AsyncClient() as httpx_client:
        agents_info = await load_all_agents(httpx_client, AGENT_ENDPOINTS)
        graph = build_workflow(agents_info).compile()

        try:
            async for message in consumer:
                question = message.value.get("q")
                session_id = message.value.get("session_id")
                
                init_state = {
                    "messages": [
                        HumanMessage(content=question)
                    ]
                }
                result = await graph.ainvoke(init_state)
                final_msg = next((m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)), "")
                print("Result:", final_msg)
                
                await producer.send_and_wait('qachat_response',{"ans": final_msg, "session_id": session_id,})
                await producer.flush()
                await consumer.commit()
        finally:
            await consumer.stop()
            await producer.stop()

if __name__ == "__main__":
    asyncio.run(run())