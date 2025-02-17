import os
import requests
from dotenv import load_dotenv

load_dotenv()

# mock - we can use some mocekd file/ endpoint like git gist
# if false we can use API call of an external service
def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    # scrape info from linkedin
    if mock:
        linkedin_profile_url = os.getenv("TEST_JSON_URL")
        response = requests.get(
            linkedin_profile_url,
            timeout=10
        )
    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.getenv("SCRAPIN_API_KEY"),
            "linkedInUrl": os.getenv("LINKEDIN_PROFILE_URL"),
        }
        response = requests.get(
            api_endpoint,
            params=params,
            timeout=10
        )
    data = response.json().get("person")
    # clean the data, some fields in the response json are empty arrays, mindful of token limits of LLMs
    # remove redundant fields
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["certifications"]
    }
    return data


# use mock = True if dont want to call API on each run . limits
if __name__ == "__main__":
    print(scrape_linkedin_profile(os.getenv("LINKEDIN_PROFILE_URL"), mock=True))