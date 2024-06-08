import os
from supabase import create_client , Client
import supabase
from groq import Groq


def generate_job_description(resume: str) -> str:
    """
    This function uses the Hyde technique where we take a candidates resume and generate a job description for which he will be suiable for.
    """

    client = Groq(
        api_key=os.environ.get("GROQ_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "The candidate has a background in machine learning. The candidates resume will be provided. Please generate a job description for the candidate."
            },
            {
                "role": "user",
                "content": "The candidate resume is: \n\n " + resume
            }
        ],
        model="llama3-70b-8192",
    )

    return chat_completion.choices[0].message.content



