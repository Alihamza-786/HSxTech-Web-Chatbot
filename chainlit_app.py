import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

import chainlit as cl
from langGraph_app import agent
from langchain_core.messages import HumanMessage, AIMessage

# -------------------------------
# Global executor and semaphore
# -------------------------------
executor = ThreadPoolExecutor(max_workers=4)  # for CPU-heavy tasks
semaphore = asyncio.Semaphore(4)             # max 4 users concurrently

async def run_in_thread(fn, *args):
    """Run blocking function in a separate thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args))

# -------------------------------
# Starters
# -------------------------------
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="🛠️ HSxTech Services", message="What Are The Services HSxTech Providing?"),
        cl.Starter(label="☸️ What is Odoo?", message="What is Odoo?"),
        cl.Starter(label="🌐 What Language Odoo is Based on?", message="What Language Odoo is Based on?"),
        cl.Starter(label="📊 What is WHT in Odoo?", message="What is WHT(Witholding Tax Configuration) in Odoo?"),
        cl.Starter(label="♾️ Odoo is Like SAP?", message="Odoo is Like SAP?")
    ]

# -------------------------------
# Chat session start
# -------------------------------
@cl.on_chat_start
async def on_chat_start():
    thread_id = str(uuid.uuid4())
    cl.user_session.set("state", {"messages": [], "thread_id": thread_id})

# -------------------------------
# Chat message handler
# -------------------------------
@cl.on_message
async def on_message(msg: cl.Message):
    state = cl.user_session.get("state")
    thread_id = state.get("thread_id")
    state["messages"].append(HumanMessage(content=msg.content))

    # Limit concurrency
    async with semaphore:
        # Run potentially CPU-heavy task in separate thread
        final_state = await agent.ainvoke(
            state,
            {"configurable": {"thread_id": thread_id}}
        )

    # Update per-user state
    final_state["thread_id"] = thread_id
    cl.user_session.set("state", final_state)

    # Show starters after every response
    starters = await set_starters()
    await cl.Message(
        content="\n\n\n\nWhat Else Would You Like To Know?",
        actions=[
            cl.Action(
                name=f"starter_{i}",
                value=starter.message,
                label=starter.label,
                payload={"message": starter.message}
            ) for i, starter in enumerate(starters)
        ]
    ).send()

# -------------------------------
# Starter actions
# -------------------------------
for i in range(5):
    @cl.action_callback(f"starter_{i}")
    async def handle_starter(action: cl.Action, i=i):
        await on_message(cl.Message(content=action.payload["message"]))

# -------------------------------
# Chat stop & end
# -------------------------------
@cl.on_stop
def on_stop():
    print("\nThe user wants to stop the task!")

@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")
