from semantic_kernel.agents import ChatCompletionAgent

def create_writer_agent(kernel):
    return ChatCompletionAgent(
        kernel=kernel,
        name="Writer",
        instructions="""
Your sole responsibility is to rewrite content according to review suggestions.
- Always apply all review directions.
- Always revise the content in its entirety without explanation.
- Never address the user.
""",
    )
