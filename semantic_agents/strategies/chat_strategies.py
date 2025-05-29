from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.contents import ChatHistoryTruncationReducer

def create_strategies(kernel, reviewer, writer):
    termination_keyword = "yes"

    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Examine the provided RESPONSE and choose the next participant.
State only the name of the chosen participant without explanation.
Never choose the participant named in the RESPONSE.

Choose only from these participants:
- {reviewer.name}
- {writer.name}

Rules:
- If RESPONSE is user input, it is {reviewer.name}'s turn.
- If RESPONSE is by {reviewer.name}, it is {writer.name}'s turn.
- If RESPONSE is by {writer.name}, it is {reviewer.name}'s turn.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=f"""
Examine the RESPONSE and determine whether the content has been deemed satisfactory.
If the content is satisfactory, respond with a single word without explanation: {termination_keyword}.
If specific suggestions are being provided, it is not satisfactory.
If no correction is suggested, it is satisfactory.

RESPONSE:
{{{{$lastmessage}}}}
""",
    )

    reducer = ChatHistoryTruncationReducer(target_count=1)

    selection_strategy = KernelFunctionSelectionStrategy(
        initial_agent=reviewer,
        function=selection_function,
        kernel=kernel,
        result_parser=lambda result: str(result.value[0]).strip() if result.value[0] else writer.name,
        history_variable_name="lastmessage",
        history_reducer=reducer,
    )

    termination_strategy = KernelFunctionTerminationStrategy(
        agents=[reviewer],
        function=termination_function,
        kernel=kernel,
        result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
        history_variable_name="lastmessage",
        maximum_iterations=10,
        history_reducer=reducer,
    )

    return selection_strategy, termination_strategy
