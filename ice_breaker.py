from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# information to check - any website source, PDF, text file or any input...
information = """
Elon Reeve Musk (/ˈiːlɒn mʌsk/; born June 28, 1971) is a businessman and U.S. special government employee, best known for his key roles in Tesla, Inc. and SpaceX, and his ownership of Twitter. Musk is the wealthiest individual in the world; as of February 2025, Forbes estimates his net worth to be US$397 billion.

A member of the wealthy South African Musk family, Musk was born in Pretoria before immigrating to Canada, acquiring its citizenship. He moved to California in 1995 to attend Stanford University, and with his brother Kimbal co-founded the software company Zip2, that was later acquired by Compaq in 1999. That same year, Musk co-founded X.com, a direct bank, that later formed PayPal. In 2002, Musk acquired U.S. citizenship, and eBay acquired PayPal. Using the money he made from the sale, Musk founded SpaceX, a spaceflight services company, in 2002. In 2004, Musk was an early investor in electric vehicle manufacturer Tesla and became its chairman and later CEO. In 2018, the U.S. Securities and Exchange Commission (SEC) sued Musk, alleging he falsely announced that he had secured funding for a private takeover of Tesla, stepped down as chairman, and paid a fine. In 2022, he acquired Twitter, and rebranded the service as X the following year.

His political activities and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation, affirming antisemitic and transphobic comments, and promoting conspiracy theories. His acquisition of Twitter was controversial due to an increase in hate speech and the spread of misinformation on the service. Musk has engaged in political activities in several countries, including as a vocal and financial supporter of U.S. president Donald Trump. He was the largest donor in the 2024 United States presidential election, and is a supporter of far-right activists, causes, and political parties. In January 2025, Musk was appointed leader of Trump's newly restructured Department of Government Efficiency.
"""

if __name__ == '__main__':
    print("Hello Langchain")

    # {} is param for the prompt
    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    #llm = ChatOpenAI(temperature=0, tiktoken_model_name="gpt-4o-mini")
    llm = ChatOllama(model="deepseek-r1:32b")

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": information})

    print(res)