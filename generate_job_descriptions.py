import os
from supabase import create_client , Client
import supabase
from groq import Groq
import pandas as pd

def generate_job_description(role , company , location , salary) -> str:
    """
    This function generates a job description given the role name, company name, pay , location and contract type. 
    """

    client = Groq(
        api_key=os.environ.get("GROQ_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You will be given a role name , company name , salary range , and location. Please generate a job description for the candidate. Specify the years of experience and the skills required for the job."
            },
            {
                "role": "user",
                "content": f"Role - {role} \n Company - {company} \n Salary - {salary} \n Location - {location}"
            }
        ],
        model="llama3-70b-8192",
    )

    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    df = pd.read_csv("./data/supabase_jobs.csv")
    output = []
    role = df['Role']
    company = df['Company']
    location = df['Location']
    salary = df['Salary']
    c = 0
    for i , j , k , l in zip(role , company , location , salary):
        print(c+1)
        output.append(generate_job_description(i , j , k , l))
        c = c + 1


    df['Job_Description'] = output
    df.to_csv("./data/job_description.csv" , index=False)