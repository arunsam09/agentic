import asyncio
import os

from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from agents import create_agents
from strategies import create_strategies


def create_kernel() -> Kernel:
    """Creates a Kernel instance with an Azure OpenAI ChatCompletion service."""
    kernel = Kernel()
    kernel.add_service(service=AzureChatCompletion())  # Configure as needed
    return kernel


async def main():
    kernel = create_kernel()

    # Create Reviewer and Writer agents
    reviewer, writer = create_agents(kernel)

    # Define strategies for selection and termination
    selection_strategy, termination_strategy = create_strategies(kernel, reviewer, writer)

    chat = AgentGroupChat(
        agents=[reviewer, writer],
        selection_strategy=selection_strategy,
        termination_strategy=termination_strategy,
    )

    print(
        "Ready! Type your input, or 'exit' to quit, 'reset' to restart the conversation. "
        "You may pass in a file path using @<path_to_file>."
    )

    while True:
        print()
        user_input = input("User > ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            await chat.reset()
            print("[Conversation has been reset]")
            continue

        if user_input.startswith("@") and len(user_input) > 1:
            file_path = os.path.join(os.path.dirname(__file__), user_input[1:])
            if not os.path.exists(file_path):
                print(f"Unable to access file: {file_path}")
                continue
            with open(file_path, encoding="utf-8") as file:
                user_input = file.read()

        await chat.add_chat_message(message=user_input)

        try:
            async for response in chat.invoke():
                if response is None or not response.name:
                    continue
                print()
                print(f"# {response.name.upper()}:
{response.content}")
        except Exception as e:
            print(f"Error during chat invocation: {e}")

        chat.is_complete = False


if __name__ == "__main__":
    asyncio.run(main())
