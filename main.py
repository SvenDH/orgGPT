from tempfile import TemporaryDirectory

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import LlamaCpp
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.agents import AgentExecutor

from llm import MistralInstruct
from tools import ToolCalling
from agent import Agent

working_directory = TemporaryDirectory()

set_llm_cache(InMemoryCache())

MODEL_PATH = "C:\\Users\\denha\\text-generation-webui\\models\\openhermes-2.5-mistral-7b\\openhermes-2.5-mistral-7b.Q4_K_M.gguf"
MODEL_PATH = "C:\\Users\\denha\\text-generation-webui\\models\\mistral-7b-instruct\\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = MistralInstruct(
    llm=LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=33,
        n_batch=512,
        n_ctx=4096,
        f16_kv=True,
        verbose=False,
        max_tokens=1024
    )
)

tool_calling_model = ToolCalling(model=llm)

search_tool = DuckDuckGoSearchRun()

tools = FileManagementToolkit(
    root_dir="data", #str(working_directory.name),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools() + [search_tool]

"""
agent_obj = Agent.from_llm_and_tools(
    model=tool_calling_model, 
    tools=tools,
)

agent = AgentExecutor.from_agent_and_tools(
    agent=agent_obj, 
    tools=tools, 
    verbose=True,
)

agent.run("Look up the latest advancements in AI, write a blog post about it and store in a file named 'blog_text.txt'")
"""


from crewai import Agent, Task, Crew

researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting
  actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=tool_calling_model
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for
  your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  tools=tools,
  llm=tool_calling_model
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Your final answer MUST be a full analysis report""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full blog post of at least 4 paragraphs.
  Write the blog post in a file named 'blog_text.txt'""",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)