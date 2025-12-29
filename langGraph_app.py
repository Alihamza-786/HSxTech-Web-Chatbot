import asyncio
import chainlit as cl

from typing import TypedDict, List, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from similarity import similarity_search
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", streaming= True)

checkpointer = MemorySaver()

# Initialize Tavily search client
tavily = TavilySearch(max_results=2)

@tool
def google_search(query: str) -> str:
    """it is used to do google search"""
    print("-------Google Search-------")
    results = tavily.run(query)
    return results


tools = [google_search, similarity_search]

llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Separated streaming function
async def stream_llm_response(messages: List[BaseMessage]) -> AIMessage:
    msg = cl.Message(content="")  # Chainlit message for streaming
    await msg.send()

    content = ""
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                content += chunk.content
                await msg.stream_token(chunk.content)
    except asyncio.CancelledError:
        print("\n⚠️ Streaming was interrupted.")
        pass

    await msg.update()
    return AIMessage(content=content)

def agent(state: AgentState):
    """Use to call tool"""
    print("\n -----------------AGENT---------------")
    messages = state["messages"]
    system_prompt = f"""
    You are a helpful assistant that provides clear, accurate, and concise answers.

    Tool usage rules:
    - If the user asks about **hsxTech**, first call `similarity_search`.  
    - If similarity_search results do not clearly answer the user’s query, then call `google_search`.
    - For **Odoo-related** questions:
    - First use `similarity_search`.
    - If the results are not clearly relevant, then use `google_search`.  
    - Always prefer answers from Odoo’s official documentation if multiple sources appear.  
    - Mention references in the final response.
    - For all **other queries**:
    - If you are unsure of the answer, call `google_search`.
    - Otherwise, answer directly without tools.

    Important:
    - Call `google_search` **at most once per user question**.
    - Call `similarity_search` **at most once per user question**.
    - After a tool returns results, use them to answer — do not call another tool for the same query.
    - Never invent or guess information.
    - Keep final responses concise (3–4 lines).
    """

    response = llm_with_tools.invoke(messages + [SystemMessage(content=system_prompt)])
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tool_call", "no_tool_call"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_call"
    return "no_tool_call"

# LangGraph node
async def generate_final_ans(state: AgentState) -> AgentState:
    messages = state["messages"]


    prompt = f"""
    You are a precise and helpful AI assistant for **HSxTech** HSxTech is a specializing in Odoo ERP implementation, customization, and 
    integrated business solutions.  
    Your job is to provide **clear, user-friendly answers in markdown format** and give answer in very simple language even non technical person can understand like CEO's of company.

    ### Context:
    {messages}

    ### Instructions:
    1. Use **only the provided Context** above to answer the user's most recent question.  
    2. If the answer is **not found in the Context**, tell him politely to ask odoo related or hsxTech related queries  
    3. Format your response in **clean markdown** (headings, bullet points, or code blocks if relevant). Do not user too big heading user h2 heading instead of h1.  
    4. Be concise but helpful. Do **not** invent, assume, or add information beyond the Context.  
    5. If the question is Odoo-related and multiple websites appear in reference, always prefer answers from Odoo’s official site, always use reference in response.
    """

    response = await stream_llm_response(prompt)
    # response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    return state


# Build LangGraph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent)
graph_builder.add_node("generate_final_ans", generate_final_ans)

tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tool_call": "tools",
        "no_tool_call": "generate_final_ans",
    }
)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("generate_final_ans", END)

agent = graph_builder.compile(checkpointer=checkpointer)