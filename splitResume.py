import os

from groq import Groq

def get_response(resume):
    client = Groq(
        api_key=os.environ.get("GROQ_KEY"),
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You will be given a resume and you need to split it into different sections like education, experience, skills, etc. The resume is given below:\n\n{resume} ",
            }
        ],
        model="llama3-70b-8192",
    )

    output = chat_completion.choices[0].message.content
    return output
