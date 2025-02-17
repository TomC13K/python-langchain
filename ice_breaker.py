import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == '__main__':
    load_dotenv()
    summary_template = """
        given the Linkedin information {information} about a person, I want you to create:
        1. a short summary
        2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    #llm = ChatOllama(model="deepseek-r1:32b")
    llm = ChatOllama(model="llama3.1")

    chain = summary_prompt_template | llm
    # use mock True if got scraping API limits
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=os.getenv("LINKEDIN_PROFILE_URL"), mock=True)
    res = chain.invoke(input={"information": linkedin_data})

    print(res)