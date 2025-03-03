from dotenv import load_dotenv

load_dotenv()
from langchain_ollama import ChatOllama
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from tools.tools import get_profile_url_tavily


def lookup(name: str) -> str:
    # llm = ChatOpenAI(
    #     temperature=0,
    #     model_name="gpt-3.5-turbo",
    # )

    llm = ChatOllama(model="llama3.1")

    template = """
       given the name {name_of_person} I want you to find a link to their Twitter profile page, and extract from it their username
       In Your Final answer only the person's username"""
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    FORMAT_INSTRUCTIONS = """Please use the following format only when you need to use a tool:
        '''
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        '''
        When you have gathered all the information regarding the user's query,\
        use the following format to answer the query and do not repeat yourself.

        '''
        Thought: Do I need to use a tool? No
        AI: [print answer and stop output]
        '''
        """

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url_tavily,
            description=f" {FORMAT_INSTRUCTIONS} useful for when you need get the Twitter Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True, handle_parsing_errors=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    twitter_username = result["output"]
    return twitter_username


if __name__ == "__main__":
    print(lookup(name="Elon Musk"))