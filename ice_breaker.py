import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.twitter import scrape_user_tweets

def ice_break_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username)

    twitter_username = twitter_lookup_agent(name=name)
    tweets = scrape_user_tweets(username=twitter_username)


    summary_template = """
            given the inforamtion about a person from Linkedin {information},
             and twitter posts {twitter_posts}, I want you to create:
            1. a short summary
            2. two interesting facts about them
            
            Use both infromations from Linkedin and twitter
        """
    summary_prompt_template = PromptTemplate(input_variables=["information", "twitter_posts"], template=summary_template)

    # llm = ChatOllama(model="deepseek-r1:32b")
    llm = ChatOllama(model="llama3.1")

    chain = summary_prompt_template | llm
    res = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})
    print(res)


if __name__ == '__main__':
    load_dotenv()

    print("Ice Breaker enter")
    ice_break_with(name="Eden MArco")
