import os
from supabase import create_client , Client
import supabase

def setup():
    client = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY")
    )
    return client

from groq import Groq

def retrieve_response(col_name: str , table_name: str , client , col_value):
    """
    Retrieve response from supabase database using the id and table_name
    """    
    data, count = client.table(table_name).select('*').eq(col_name , col_value).execute()
    return data

def update_response(col_name , col_value ,  table_name: str , client):
    """
    Update response in the supabase database using the id and table_name
    """
    data, count = client.table(table_name).update({col_name : col_value}).eq('id', 1).execute()
    return data

def get_response(resume , file_name):
    """
    resume : text of resume
    file_name : will be the key
    """

    #get supanase client for pushing data into the database
    supabase_client = setup()

    #get the response from the groq api for LLMs
    client = Groq(
        api_key=os.environ.get("GROQ_KEY"),
    )

    table_name = os.environ.get("SUPA_BASE_TABLE")
    
    schema = {
        "type":"object",
        "properties": {
            "name": {"type": "string",
                     "description": "The name of the candidate"
                     },
            "contact_information": {"type": "string" ,
                                    "description": "The contact information of the candidate. This should include the email address and phone number of the candidate. If the information is not present in the resume, just mention that the information is not present."
                                    },
            "summary": {"type": "string",
                        "description": "The summary of the candidate. This should include a brief overview of the candidate's background and experience. If the information is not present in the resume, just mention that the information is not present."
                        },
            "work_experience": {"type": "string" , 
                                "description": "The work experience of the candidate"
                                },
            "education": {"type": "string",
                          "description": "The education of the candidate"
                          },
            "skills": {"type": "string",
                       "skills": "The skills of the candidate"
                       },
            "certifications": {"type": "string",
                               "description": "The certifications of the candidate"
                               },
                               
        },
    }
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"""You are a recruiter and you have received a resume from a candidate. You need to split the resume into the following sections:\n
                1. Name\n
                2. Contact Information\n
                3. Summary\n
                4. Work Experience\n
                5. Education\n
                6. Skills\n
                7. Certifications\n

                If the information is not present in the resume, just mention that the information is not present.

                Return a JSON object with the following schema: 
                {schema}
                .
                """,
            },
            {
                "role": "user",
                "content": f"The Resume is: \n\n{resume} ",
            }
        ],
        model="llama3-70b-8192",
        response_format={"type": "json_object"}
    )

    output = chat_completion.choices[0].message.content

    #process output as dict 
    output_dict = eval(output)


    name = output_dict['name']
    contact_information = output_dict['contact_information']
    summary = output_dict['summary']
    work_experience = output_dict['work_experience']
    education = output_dict['education']
    skills = output_dict['skills']
    certifications = output_dict['certifications']

    st = ""
    st = st + "Name: " + name + "\n\n"
    st = st + "Contact Information: " + contact_information + "\n\n"
    st = st + "Summary: " + summary + "\n\n"
    st = st + "Work Experience: " + work_experience + "\n\n"
    st = st + "Education: " + education + "\n\n"
    st = st + "Skills: " + skills + "\n\n"
    st = st + "Certifications: " + certifications + "\n\n"

    #replace space in file name with underscore
    file_name = file_name.replace(" ", "_")
    file_name = file_name.split(".pdf")[0]
    new_name = name.replace(" ", "_")


    #push the response to the database
    response = {
        "id": 1 , 
        "resume_key": file_name + "_" + new_name,
        "name": name,
        "resume_content": st,
    }

    #delete the previous record with the same id
    data, count = supabase_client.table(table_name).delete().eq('id', 1).execute()

    #insert the new record
    data, count = supabase_client.table(table_name).insert(response).execute()
    
    #retrieve the response from the database
    print("retrieve response")
    data_retrieved = retrieve_response(col_name='id' , table_name=table_name , client=supabase_client , col_value=1)
    
    #get conversation from the response
    conversation = data_retrieved[1][0]['conversation'] 
    
    return st
