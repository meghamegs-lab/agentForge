"""
Chainlit chat UI for Fortio - the Ghostfolio Finance Agent.
Run with: chainlit run agent/ui/chainlit_app.py
"""
import uuid
import chainlit as cl
from langchain_core.messages import HumanMessage

from agent.graph.graph import agent_graph
from agent.graph.state import AgentState


@cl.on_chat_start
async def on_chat_start():
    """Initialize session with a unique conversation ID."""
    session_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", session_id)
    cl.user_session.set("messages", [])

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
    """Handle incoming user message through the agent graph."""
    conversation_id = cl.user_session.get("conversation_id")
    messages = cl.user_session.get("messages", [])

    # Add user message
    messages.append(HumanMessage(content=message.content))

    # Show thinking indicator
    thinking_msg = cl.Message(content="")
    await thinking_msg.send()

    try:
        # Build initial state
        state: AgentState = {
            "messages": messages,
            "tool_results": [],
            "verification_flags": [],
            "confidence": "HIGH",
            "reasoning_steps": 0,
            "conversation_id": conversation_id,
            "user_id": "demo_user",
            "final_response": "",
            "should_escalate": False,
        }

        # Stream through the graph
        final_state = await agent_graph.ainvoke(state)

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
        full_response = response_text + footer

        # Update thinking message with response
        thinking_msg.content = full_response
        await thinking_msg.update()

        # Update session message history
        messages = list(final_state["messages"])
        cl.user_session.set("messages", messages)

    except Exception as e:
        thinking_msg.content = (
            f"❌ Sorry, I encountered an error: {str(e)}\n\n"
            "Please try again or rephrase your question."
        )
        await thinking_msg.update()
