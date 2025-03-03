import os
from typing import Tuple

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.twitter import scrape_user_tweets
from output_parsers import summary_parser, Summary

def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=twitter_username)


    summary_template = """
            given the information about a person from Linkedin {information},
             and twitter posts {twitter_posts}, I want you to create:
            1. a short summary
            2. two interesting facts about them
            
            Use both information from Linkedin and twitter
            \n{format_instructions}
        """
    # partial variables, pluggin in advance of the prompt template, things we know we want to invoke
    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )

    # llm = ChatOllama(model="deepseek-r1:32b")
    llm = ChatOllama(model="llama3.1")

    #chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # langchain expression language - for writting chains
    chain = summary_prompt_template | llm | summary_parser
    res: Summary = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})

    return res, linkedin_data.get("profile_pic_url")


if __name__ == '__main__':
    load_dotenv()

    print("Ice Breaker enter")
    ice_break_with(name="Eden MArco")
