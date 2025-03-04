from typing import Union, List

from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_ollama import ChatOllama

from callbacks import AgentCallbackHandler


# Tool decorator will take this function and create a tool from it
# rakes function signatures and populate the tools agent class
# - description is used by the LLM
# -executes as any other agent took using invoke() and consumes dictionary
@tool
def get_text_length(text: str) -> int:
    """Returns the length of the character"""
    print(f"get_text_length enter with {text}")
    # strip away non alphanumeric chars
    text = text.strip("'n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[tool], tool_name: str) -> tool:
    for t in tools:
        if t.name == tool_name:
            return tool
    raise ValueError(f"No tool found with name {tool_name}")


if __name__ == '__main__':
    print("hello react langchain")
    tools = [get_text_length]  # list of tools that will be provided to react agents

    # custom prompt from langchain
    # action - is action to take and should be one of the tool names (get_text_length)
    # action  input - input for that action (input for the get_text_length function)
    # observation - result of the action - result of the tool

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """
    # agent_scratchpad - will contain all the history and all the information in this react execution

    # in partial we know we want to initialize the get_length method and input sofr it
    # tool_names will be list of comma separated names
    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools),
                                                                     tool_names=", ".join(
                                                                         [t.name for t in tools]),
                                                                     )

    # temp is zero so we dont have any creative answers
    # stop argument - valueis Observation to stop generating words once it generates \nObservation token
    # callbacks call our custom logger to display all calls and responses in llm
    llm = ChatOllama(temperature=0, stop=["\nObservation"], model="llama3.1", callbacks=[AgentCallbackHandler])
    intermediate_steps= []


    # LCEL Langchain expression language - declare declaratively and compose chains together, using pipes
    # pipe operator takes output of the left side and plugs to input to the right side
    # so we take prompt n run it on llm input
    # dictionary with key input provides question we asking our agent
    # to make it dynamic and we want to invoke when the chain will be invoked, we want to supply it with
    # dictionatry of the keywords, we replace in placeholders - lambda function, that receives dictionary and accessing its input key
    # ReActSingleInputOutputParser - output parsing - parse output from LLM into a structured format
    # format_log_to_str - parse only text so LLm doesnt have any problems
    agent = ({
                "input": lambda x: x["input"],
             "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
             }
             | prompt
             | llm
             | ReActSingleInputOutputParser()
             )

    agent_step = ""
    # loop to run until we get response of type AgentFinish
    while not isinstance(agent_step, AgentFinish):

    # agent reasoning
    # res = agent.invoke({
    #     "input": "What is the length in characters of the text DOG? ",
    #     "agent_scratchpad": intermediate_steps,
    # }
    # )
    # print(res)

        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the  length of 'DOG' in characters? ",
             "agent_scratchpad": intermediate_steps,
             })
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f'{observation}')
            # appanding history of what was already performed and observed
            intermediate_steps.append((agent_step, str(observation)))

    # dont need 2nd invoke when we have a loop
    # agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
    #     {
    #         "input": "What is the  length of 'DOG' in characters? ",
    #         "agent_scratchpad": intermediate_steps,
    #     })
    # output parsing returns or AgentAction or AgentFinish
    # it returns AgentFinish because the previous log was already final answer where llm got correct answer
    # print(agent_step)

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
