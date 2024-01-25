from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import LlamaCpp
from langchain_community.agent_toolkits import FileManagementToolkit
from crewai import Task, Crew

from llm import MistralInstruct, MistralChat, ZephyrChat
from agent import StructuredAgent as Agent


MODEL_PATH = "C:\\Users\\denha\\text-generation-webui\\models\\openhermes-2.5-mistral-7b\\openhermes-2.5-mistral-7b.Q4_K_M.gguf"
#MODEL_PATH = "C:\\Users\\denha\\text-generation-webui\\models\\mistral-7b-instruct\\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

#MODEL_PATH = "D:\\Downloads\\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
#MODEL_PATH = "D:\\Downloads\\openhermes-2.5-mistral-7b.Q4_K_M.gguf"
MODEL_PATH = "D:\\Downloads\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


llm = ZephyrChat(
    llm=LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=33,
        n_batch=512,
        n_ctx=4096,
        f16_kv=True,
        max_tokens=1024,
        #callback_manager=callback_manager,
        verbose=False,
    )
)

search_tool = DuckDuckGoSearchRun()

tools = FileManagementToolkit(
    root_dir="data", #str(working_directory.name),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools() + [search_tool]


researcher = Agent(
  role='Senior Research Analyst',
  goal='Searching the internet, comprehending details, and finding information.',
  backstory="""You are an advanced web information retriever. You will receive a goal and need to perform research to answer it.
      1. You **MUST** first plan your research.

      2. For each step, you will web search for results. You can perform queries in parallel.

        Do NOT perform yearly individual searches unless absolutely required. This wastes resources and time. Always aim for consolidated data over a range of years.

        Example of undesired behavior: Searching "US births 2019", then "US births 2020", then "US births 2021"...
        Desired behavior: Searching "US births from 2019 to 2021"

      3. If by searching for something specific you find something else that is relevant, state it and consider it.

      4. If the research verification says the data is incomplete, search for the missing data. If you still cannot find it, consider it unavailable and don't fail; just return it.

      5. Use scrape_text for getting all the text from a webpage, but not for searching for specific information.
      
      6. RESPECT USER'S DESIRED FORMAT""",
  verbose=True,
  allow_delegation=False,
  tools=tools,
  llm=llm
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
  llm=llm
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