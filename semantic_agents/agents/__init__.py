from .reviewer import create_reviewer_agent
from .writer import create_writer_agent

def create_agents(kernel):
    return create_reviewer_agent(kernel), create_writer_agent(kernel)
