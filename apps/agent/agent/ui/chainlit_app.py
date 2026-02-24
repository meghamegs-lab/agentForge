"""
Chainlit chat UI for Fortio - the Ghostfolio Finance Agent.
Run with: chainlit run agent/ui/chainlit_app.py

Checkpointing:
  Uses the module-level `agent_graph` singleton which is compiled with
  MemorySaver. History persists for the lifetime of the Chainlit server
  process. Each browser session gets its own `conversation_id` (thread_id),
  so users are fully isolated from each other.
"""
import uuid
import chainlit as cl
from langchain_core.messages import HumanMessage

from agent.graph.graph import agent_graph


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new session with a unique conversation ID (LangGraph thread_id)."""
    session_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", session_id)
    # No need to track messages manually — MemorySaver handles it via thread_id.

    await cl.Message(
        content=(
            "👋 **Welcome to Fortio, your Ghostfolio Finance Assistant!**\n\n"
            "I can help you understand your investment portfolio. Try asking:\n"
            "- *What does my portfolio look like?*\n"
            "- *How has my portfolio performed this year?*\n"
            "- *Am I too concentrated in any sector?*\n"
            "- *What are my biggest fees this year?*\n"
            "- *What's the current price of AAPL?*\n\n"
            "⚠️ I provide information for educational purposes only — not financial advice."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user message through the agent graph.

    Only the NEW message is passed in state — MemorySaver automatically
    loads and merges previous messages via thread_id.
    """
    conversation_id = cl.user_session.get("conversation_id")

    # LangGraph config: thread_id tells the checkpointer which conversation
    # history to load and save to.
    config = {
        "configurable": {
            "thread_id": conversation_id,
            "user_id": "demo_user",
        }
    }

    # Show thinking indicator
    thinking_msg = cl.Message(content="")
    await thinking_msg.send()

    try:
        # Only pass the NEW message — LangGraph loads history from MemorySaver.
        # tool_results and verification_flags reset each turn (no reducer).
        state = {
            "messages": [HumanMessage(content=message.content)],
            "tool_results": [],
            "verification_flags": [],
            "confidence": "HIGH",
            "reasoning_steps": 0,
            "conversation_id": conversation_id,
            "user_id": "demo_user",
            "final_response": "",
            "should_escalate": False,
        }

        # Pass config so MemorySaver can load/save state for this thread_id
        final_state = await agent_graph.ainvoke(state, config=config)

        # Get the verified response
        response_text = final_state.get("final_response", "")
        if not response_text:
            # Fallback: get from last message
            last_msg = final_state["messages"][-1]
            if hasattr(last_msg, "content"):
                content = last_msg.content
                response_text = content if isinstance(content, str) else str(content)

        # Build confidence badge
        confidence = final_state.get("confidence", "MEDIUM")
        confidence_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "🟡")
        flags = final_state.get("verification_flags", [])

        # Append metadata footer
        footer_parts = [f"{confidence_emoji} Confidence: **{confidence}**"]
        if flags:
            high_flags = [f for f in flags if f.get("severity") == "HIGH"]
            if high_flags:
                footer_parts.append(f"⚠️ {len(high_flags)} high-severity flag(s) detected")

        footer = "  \n*" + " | ".join(footer_parts) + "*"
        thinking_msg.content = response_text + footer
        await thinking_msg.update()

    except Exception as e:
        thinking_msg.content = (
            f"❌ Sorry, I encountered an error: {str(e)}\n\n"
            "Please try again or rephrase your question."
        )
        await thinking_msg.update()
