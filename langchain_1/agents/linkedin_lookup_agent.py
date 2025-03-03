from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from tools.tools import get_profile_url_tavily

load_dotenv()


def lookup(name: str) -> str:
    # llm = ChatOpenAI(
    #     temperature=0,
    #     model_name="gpt-3.5-turbo",
    # )
    llm = ChatOllama(model="llama3.1")
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""
    # https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.agent.create_react_agent.html
    # template= '''Answer the following questions as best you can.
    #
    # Use the following format:
    #
    # Question: the input question you must answer
    # Thought: you should always think about what to do
    # Action: the action to take
    # Action Input: the input to the action
    # Observation: the result of the action
    # ... (this Thought/Action/Action Input/Observation can repeat N times)
    # Thought: I now know the final answer
    # Final Answer: the final answer to the original input question
    #
    # Begin!
    #
    # Question: Given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.Your answer should contain only a URL
    # Thought:'''

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )



    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description=f"useful for when you need get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True, handle_parsing_errors=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url

if __name__ == "__main__":
    linked_profile_url = lookup(name="Eden Marco")
    print(linked_profile_url)