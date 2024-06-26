import os
import functools
import traceback
import json
import requests
import pdb
from pprint import pprint
from typing_extensions import TypedDict
from typing import List, Any, Annotated
import io
import operator
import random

from dotenv import load_dotenv, find_dotenv
from PIL import Image
import pandas as pd
import PyPDF2
from groq import Groq
from supabase import create_client
import chromadb

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from langgraph.graph import END, StateGraph
# for RAG
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question about the candidate
        intent: predicted user's intent from action

        attachment: path to audio or pdf

        resume_key: resume key
        resume_content: resume contents

        speech_key: speech key
        speech_transcribed: audio transciption
        speech_keypoints: key information extracted from speech_transcribed

        answer: Agent's answer to question
    """
    question: str
    intent: str

    attachment: str

    resume_key: str
    resume_content: str

    speech_key: str
    speech_transcribed: dict
    speech_keypoints: dict

    answer: str
    # aggregate: Annotated[list, operator.add] # for multiple connections to/from nodes


class RecruiterAgent():
    def __init__(self) -> None:
        self.JIGSAW_API_KEY = os.environ["JIGSAW_API_KEY"]
        self.JIGSAW_PUBLIC_KEY = os.environ["JIGSAW_PUBLIC_KEY"]
        self.verbose = True
        self.resume_full = ""

        self.GROQ_LLM, self.GROQ_v2, self.OpenAI_LLM = self.load_llm_model()

        self.setup_tools()
        self.setup_chains()
        self.app = self.setup_agent()

        self.load_data()
        self.setup_supabase()
        # self.embed_JD_upload() # delete table wont affect this

        # self.warmup()
        self.verbose = False

    def warmup(self, ):
        print('--- warmup')
        try:
            _resume_parser_test = self.app.invoke({"attachment": 'data/JohnDoeResume.pdf'})
            _speech_parser_test = self.app.invoke({"attachment": 'data/Interview.wav'})
            # pdb.set_trace()
            # _speech_parser_test = self.app.invoke({"question": 'get_summary'})
            # _speech_parser_test = self.app.invoke({"question": 'what is the educationl history ?'})
            # pdb.set_trace()
        except:
            traceback.print_exc()
            pdb.set_trace()
            _ = 'warmup'

    # ------------------------------------------------------------
    # Supabase
    def setup_supabase(self, ):
        self.supabase_client = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_KEY")
        )
        self.table_name = os.environ.get("SUPA_BASE_TABLE")
        # pdb.set_trace()
        data_deleted, count = self.supabase_delete()
        data_created, count = self.supabase_insert_new_record({"id": 1 , "resume_key": "resume_key"})
        return
        pdb.set_trace()
        pprint(self.supabase_retrieve())
        self.supabase_update_column(col_name="resume_key", col_value="JohnDoeResume.pdf")
        self.supabase_update_column(col_name="resume_content", col_value="Name: JOHN DOE")
        self.supabase_update_column(col_name="conversation_key", col_value=["call_v1.wav"])
        self.supabase_update_column(col_name="conversation", col_value=["hello"])
    def supabase_delete(self, ):
        # delete the previous record with the same id
        data, count = self.supabase_client\
                        .table(self.table_name)\
                        .delete()\
                        .eq('id', 1)\
                        .execute()
        return data, count
        """
        data, count = self.supabase_client.table(self.table_name).delete().eq('id', 1).execute()

        data
            ('data', 
            [
                {   'created_at': '2024-06-08T09:05:44.470957+00:00',
                    'id': 1,
                    'name': 'JOHN DOE',
                    'resume_content': 'Name: JOHN DOE\n'
                                        '\n'...
                                        'Certifications: Salt - SaltStack Certified Engineer, GCP '
                                        '- Professional Cloud Architect\n'
                                        '\n',
                    'resume_key': 'sample_pdf_JOHN_DOE',
                    'conversation_key': [str],
                    'conversation': [str],
                }
            ]
            )
        """
    def supabase_insert_new_record(self, response_dict):
        """
        response_dict = {
            "id": 1 , 
            "resume_key": file_name + "_" + new_name,
            "name": name,
            "resume_content": st,
        }
        """
        #delete the previous record with the same id
        data, count = self.supabase_client\
                        .table(self.table_name)\
                        .insert(response_dict)\
                        .execute()
        return data, count
        """
        response_dict = {"id": 1 , "resume_key": "resume_key"}
        data, count = self.supabase_client.table(self.table_name).insert(response_dict).execute()
        """

    def supabase_retrieve(self, col_name='id', col_value=1):
        """
        Retrieve response from supabase database using the id and self.table_name
        """
        data_retrieved, count = self.supabase_client\
                                    .table(self.table_name)\
                                    .select('*')\
                                    .eq(col_name , col_value)\
                                    .execute()
        return data_retrieved[1][0]
        # example of getting conversation from the response
        conversation = data_retrieved[1][0]['conversation']
        """
        col_name='id';
        col_value=1
        data_retrieved, count = self.supabase_client.table(self.table_name).select('*').eq(col_name , col_value).execute()
        data_retrieved = data_retrieved[1][0]
        pprint(data_retrieved)
        {
            'created_at': '2024-06-08T07:42:49.651779+00:00',
            'id': 1,
            'name': 'JOHN DOE',
            'resume_content': 'Name: JOHN DOE\n'
                            ...
                            'Certifications: Salt - SaltStack Certified Engineer, GCP - '
                            'Professional Cloud Architect\n'
                            '\n',
            'resume_key': 'sample_pdf_JOHN_DOE'
            'conversation_key': ['Interview.wav'],
            'conversation': ['hi there this is leonard ....'],
            'conversation_keypoints': ['Name:..., Expected Salary: ...'],
        }
        """

    def supabase_update_column(self, col_name, col_value):
        """
        Update response in the supabase database using the id and table_name
        """
        data, count = self.supabase_client\
                        .table(self.table_name)\
                        .update({col_name : col_value})\
                        .eq('id', 1)\
                        .execute()
        return data
        """
        col_name = "conversation_key"
        col_value = ["conversation_key_v1"]
        data, count = self.supabase_client.table(self.table_name).update({col_name : col_value}).eq('id', 1).execute()

        col_name = "conversation"
        col_value = ["transcribed call 1"]
        data, count = self.supabase_client.table(self.table_name).update({col_name : col_value}).eq('id', 1).execute()
        """

    # ------------------------------------------------------------
    # Model / Data
    def load_llm_model(self, ):
        print('\n--- load_llm_model')
        model_gemma = "gemma-7b-lt"
        model_llama3_70b = "llama3-70b-8192"
        model_llama3_8b = "llama3-8b-8192"
        model_mixtral = "mixtral-8x7b-32768"
        self.model_name = model_llama3_70b

        # For LCEL
        GROQ_LLM = ChatGroq(model=self.model_name,)
        print(f'\tUsing {GROQ_LLM.model_name}')

        # Lazy to figure out how to merge these to LCEL
        GROQ_v2 = Groq(
            api_key=os.environ.get("GROQ_KEY"),
        )

        # compute job description embeddings
        OpenAI_LLM = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        return GROQ_LLM, GROQ_v2, OpenAI_LLM

    def load_data(self, ):
        self.job_description_df = pd.read_csv("./data/job_description.csv")
        """
        Role             |                                 Software Engineer
        Company          |                                   Mary Technology
        Company URL      |                   https://www.marytechnology.com/
        Salary           |               $100K - $180K (on experience/level)
        Location         |                              Sydney, NSW; REMOTE.
        Application Link | Please e-mail or Slack message Daniel Lord-Doyle.
        Type             |                                               NaN
        Email            |                            dan@marytechnology.com
        Date Created     |                                         4/21/2024
        idx              |                                                 5
        job_description  |                                               NaN
        Job_Description  | Here is a job description for a Software Engin...  <---  generated from LLM
        """
        self.job_des = self.job_description_df['Job_Description'].tolist()
    def embed_JD_upload(self, ):
        for i, job_des in enumerate(self.job_des):
            response = self.OpenAI_LLM.embeddings.create(
                input=job_des,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding
            # len = 1536 | [0.005, -0.003, 0.036, ..., 0.029, 0.003, -0.021]
            response = {
                'embedding': embedding,
                'description': job_des,
            }
            # response = { 'embedding': embedding, 'description': job_des, }
            pdb.set_trace()
            data, count = self.supabase_client\
                            .table("embeddings")\
                            .insert(response)\
                            .execute()
            # data, count = self.supabase_client.table("embeddings").insert(response).execute()
            print(f'len(data[1]) = {len(data[1])}')

    # ------------------------------------------------------------
    # RAG Database
    def RAG_setup(self):
        """
        Pseudo RAG setup
        """
        self.vectordb = None
        self.retriever = None

    def RAG_add(self, resume_or_speech, content_tuple):
        """
        Pseudo RAG update
        In reality now, we're uploading everything to supabase first
        """
        if resume_or_speech == 'resume':
            resume_key, resume_content = content_tuple
            self.supabase_update_column(col_name="resume_key", col_value=resume_key)
            self.supabase_update_column(col_name="resume_content", col_value=resume_content)
        if resume_or_speech == 'speech':
            speech_key, speech_transcribed, speech_keypoints = content_tuple
            self.update_conversation(speech_key, speech_transcribed, speech_keypoints)
        return
        """
        new_docs = Document(resume_content, metadata = {'document_type': resume_key})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        texts = text_splitter.split_documents([new_docs])
        len(texts)
        """

    def RAG_search(self, content_level):
        """
        Pseudo RAG retrieval
        In reality now, we're retrieving everything from supabase first

        content_level = {'summary', 'full'}
        """
        # retrieve resume / speech_keypoints | from supabase
        data_retrieved = self.supabase_retrieve()

        resume_full = f'Full Resume: \n```\n{self.resume_full}\n```\n'

        resume_summary = ""
        if "resume_content" in data_retrieved:
            resume_summary += f'Resume summary: \n```\n{data_retrieved["resume_content"]}\n```\n'

        conversation_full = ""
        if "conversation" in data_retrieved:
            conversation_lst = data_retrieved["conversation"]
            conversation = "\n".join(conversation_lst) if len(conversation_lst) else ''
            conversation_full += f'Past conversations (if any): \n```\n{conversation}\n```\n'

        conversation_summary = ""
        if "conversation_keypoints" in data_retrieved:
            conversation_keypoints_lst = data_retrieved["conversation_keypoints"]
            conversation_keypoints = "\n".join(conversation_keypoints_lst) if len(conversation_keypoints_lst) else ''
            conversation_summary += f'Past conversations summary (if any): \n```\n{conversation_keypoints}\n```\n'

        RAG_context = ""
        if content_level == 'full':
            RAG_context = resume_full 
            RAG_context += "\n****************************************************************\n" 
            RAG_context += conversation_full
        elif content_level == 'summary_more':
            RAG_context = resume_full 
            RAG_context += "\n****************************************************************\n" 
            RAG_context += resume_summary
            RAG_context += "\n****************************************************************\n" 
            RAG_context += conversation_summary
        elif content_level == 'summary_lite':
            RAG_context += resume_summary
            RAG_context += "\n****************************************************************\n" 
            RAG_context += conversation_summary
        
        return RAG_context

    # ------------------------------------------------------------
    # Tools (resume)
    def JIGSAW_upload_resume_file(self, resume_file):
        resume_key = os.path.basename(resume_file).replace(' ', '_')
        assert resume_key.endswith('.pdf')
        return resume_key, None
        # Define the endpoint and the API key
        #            <                endpoint               >
        #                                                      <      key     >
        #                                                                       <overwrite file>
        endpoint = f"https://api.jigsawstack.com/v1/store/file?key={resume_key}&overwrite=true"
        print(f"\tendpoint = {endpoint}")
        # Define the headers
        headers = {
            "Content-Type": "document/pdf",
            "x-api-key": self.JIGSAW_API_KEY,
        }

        # pass in the bytes of the audio and upload to the Jigsaw cloud
        body = open(resume_file, 'rb')

        # Make the POST request
        response = requests.post(endpoint,
                                 headers=headers,
                                 data=body
                                )

        # Parse the JSON response
        data = response.json()
        print(f"\tdata = {data}")

        return resume_key, data
        {'success': True, 'url': 'https://api.jigsawstack.com/v1/store/file/gettysburg.wav', 'key': 'gettysburg.wav'}

    def resume_parser_tool(self, uploaded_file):
        # uploaded_file = "./data/JohnDoeResume.pdf"
        resume_key, data_upload = self.JIGSAW_upload_resume_file(uploaded_file)
        pdfReader = PyPDF2.PdfReader(uploaded_file)
        resume_text = "\n"
        for pageObj in pdfReader.pages:
            current_text = pageObj.extract_text()
            resume_text += f"{current_text}\n"
        return resume_text, resume_key
        response = get_response(resume_text)
        display(Markdown(response))

    # ------------------------------------------------------------
    # Tools (speech)
    def JIGSAW_upload_speech_file(self, speech_file):
        speech_key = os.path.basename(speech_file).replace(' ', '_')
        assert speech_key.endswith('.wav')
        # return speech_key, None
        # Define the endpoint and the API key
        #            <                endpoint               >
        #                                                      <      key     >
        #                                                                       <overwrite file>
        endpoint = f"https://api.jigsawstack.com/v1/store/file?key={speech_key}&overwrite=true"
        print(f"\tendpoint = {endpoint}")
        # Define the headers
        headers = {
            "Content-Type": "audio/wav",
            "x-api-key": self.JIGSAW_API_KEY,
        }

        # pass in the bytes of the audio and upload to the Jigsaw cloud
        body = open(speech_file, 'rb')

        # Make the POST request
        response = requests.post(endpoint,
                                headers=headers,
                                data=body
                                )

        # Parse the JSON response
        data = response.json()
        print(f"\tdata = {data}")

        return speech_key, data
        {'success': True, 'url': 'https://api.jigsawstack.com/v1/store/file/gettysburg.wav', 'key': 'gettysburg.wav'}

    def JIGSAW_transcribe_file_v2(self):
        return {
            'success': True,
            'text': """Hi, my name is Leonard and I'm a recruiter for Good Talent Company. Hi Leonard, my name is John Doe. Hi, so I just want to run some questions through you. So may I know what's your gender? Yeah, I'm a male and I'm 26. What languages do you speak? I speak English and Mandarin. And which country are you residing in? I moved to Singapore 4 years back and I'm you residing in uh actually i moved to singapore four years back and i'm currently residing in singapore yeah okay thanks for that uh do you require a visa sponsorship uh yes yeah i do need it yeah i'm currently on an employment pass so yeah in even in the future i would need one yeah okay thanks for that uh what is your current notice period What is your current notice period? My current notice period is one month. Why are you looking out for a new job? I feel like I would like to have some better career growth opportunities and a bit more of learning as well in my current role. Good, that's great. Can you elaborate more in detail what are you looking out for in the next role? Yeah, so I would actually like to work in a in a team which has very good engineering culture where there's a lot of feedback loops from like senior engineers as well about programming and stuff like and maintaining a very good work culture and And there's an environment where like I get to learn new technological stacks as well. Interesting. Okay, so currently do you have any other interviews right now? Yeah, so as you know I am actually interviewing with two other companies, Tech Solutions and DataCorp. Okay, so what are you currently drawing now for your salary? Oh yeah, so mine is approximately $80,000 per year without the bonus and one year month-on-month bonus based on performance. Okay, what is your expected salary? Yeah, I would actually, based on the current industry trends, I would like my current salary to be around 100,000 Singapore dollars a year. Okay, sure. So, just to fill out some of the background check, may I know what's your identity number? Yeah, so my number is G1234567N. Okay. Will you be open to a consulting model? Yeah, definitely. Yeah. Okay. okay currently how many leave do you have oh yeah so I have 18 days of holiday leave and 30 days of sick leave yeah have you worked for this company good startup before no I haven't there's the first time I'm in fact up like okay okay so yeah I would like to I think that you're a good match so I would like to recommend that for you. So any final comments? Actually I'm highly motivated and I feel like I would be a perfect fit for this company because I've looked at some of your teams and the kind of work that they do. So I'm definitely interested in interviewing for this particular role. Okay thank you very much. I appreciate your time. I'll let you know if I have any updates for you. Yeah sure thank you and I look forward to hearing from you.
            """,
            'chunks': [
            ]
        }
    def JIGSAW_transcribe_file(self, speech_key):
        # Define the endpoint and the API key
        endpoint = "https://api.jigsawstack.com/v1/ai/transcribe"
        print(f"\tendpoint = {endpoint}")

        # Define the headers
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.JIGSAW_API_KEY,
        }

        # Define the body of the request
        body = {
            "file_store_key": speech_key,
            "language": "en",
            "translate":True,
        }
        print(f"\tbody = {body}")

        # Make the POST request
        print(f"\tRunning real tanscription now")
        response = requests.post(endpoint,
                                headers=headers,
                                data=json.dumps(body)
                                )

        # Parse the JSON response
        data = response.json()

        # Print the result
        print(f"\tdata = {data}")
        return data
        {
            'success': True,
            'text': ' four score and seven years ago our fathers brought forth on this continent a new nation conceived in liberty and dedicated to the proposition that all men are created equal now we are engaged in a great civil war testing whether that nation or any nation so conceived and so dedicated can long endure',
            'chunks': [
                {
                    'timestamp': [0, 17.58],
                    'text': ' four score and seven years ago our fathers brought forth on this continent a new nation conceived in liberty and dedicated to the proposition that all men are created equal now we are engaged in a great civil war testing whether that nation or any nation so conceived and so dedicated can long endure'
                }
            ]
        }

    def speech_parser_tool(self, speech_file):
        # upload file
        # speech_key, data_upload = self.JIGSAW_upload_speech_file(speech_file)
        speech_key = "Interview.wav"

        # transcribe
        print(f'speech_key = {speech_key}')
        if speech_key == 'Interview.wav':
            speech_transcribed = self.JIGSAW_transcribe_file_v2()
        else:
            pdb.set_trace()
            speech_transcribed = self.JIGSAW_transcribe_file(speech_key)
        return speech_key, speech_transcribed

    # ------------------------------------------------------------
    # Tools
    def setup_tools(self):
        return

    def invoke_with_retry(self, chain, input_data, max_retries=3):
        for attempt in range(max_retries):
            try:
                result = chain.invoke(input_data)
                return result
            except ValueError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                    """
                    Gracefully fail here
                    """

    def process_xml_tags(self, input_text, tag):
        try:
            extracted_content, _ = input_text.split(f'</{tag}>', 1)
            _, extracted_content = extracted_content.split(f'<{tag}>', 1)
            return extracted_content
        except:
            traceback.print_exc()
            print(input_text)
            return input_text

    # ------------------------------------------------------------
    # Chains
    def _chain_resume_parser(self, resume_text):
        print('\t _chain_resume_parser')
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
        chat_completion = self.GROQ_v2.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are a recruiter and you have received a resume from a candidate.
You need to split the resume into the following sections:\n
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
                    "content": f"The Resume is: \n\n{resume_text} ",
                }
            ],
            model=self.model_name, # "llama3-70b-8192", 
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

        return st

    def _chain_speech_parser(self, speech_transcribed):
        print('\t _chain_speech_parser')
        print(f'\t speech_transcribed = {speech_transcribed}')

        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the candidate"
                },
                "spoken_languages": {
                    "type": "list",
                    "items": {
                        "type": "string"
                    },
                    "description": "All the languages the candidate can speak in"
                },
                "country_residence": {
                    "type": "string",
                    "description": "Which country the candidate is residing at now"
                },
                "work_pass_type": {
                    "type": "string",
                    "description": "The type of work pass the candidate has (choose from 'visa', 'permanent_resident', 'citizen', 'not mentioned')"
                },
                "nationality": {
                    "type": "string",
                    "description": "The candidate's nationality"
                },
                "notice_period": {
                    "type": "string",
                    "description": "The amount of notice the candidate needs to give their current employer"
                },
                "reason_for_looking_out": {
                    "type": "string",
                    "description": "Why the candidate is seeking a new job"
                },
                "looking_for_in_next_move": {
                    "type": "string",
                    "description": "What the candidate is seeking in their next job/company"
                },
                "other_interviews_in_pipeline": {
                    "type": "string",
                    "description": "Other companies the candidate is interviewing with"
                },
                "current_salary": {
                    "type": "string",
                    "description": "The candidate's current salary"
                },
                "expected_salary": {
                    "type": "string",
                    "description": "The salary the candidate is seeking"
                },
                "identity_number": {
                    "type": "string",
                    "description": "The candidate's NRIC, FIN, or Passport number (for permanent roles only)"
                },
                "open_to_consulting_model": {
                    "type": "string",
                    "description": "Whether the candidate is open to a consulting arrangement (for contract roles only)"
                },
                "current_annual_leave_days": {
                    "type": "string",
                    "description": "The candidate's current number of annual leave days (for contract roles only)"
                },
                "worked_for_client_before": {
                    "type": "string",
                    "description": "Whether the candidate has worked for the client company before ('yes' or 'no')"
                },
                "general_notes": {
                    "type": "string",
                    "description": "Any other relevant / miscellanous information about the candidate"
                }
            }
        }

        chat_completion = self.GROQ_v2.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are an AI assistant that specializes in summarizing conversations and extracting key information.
You will be provided with the transcript of an audio conversation between a job candidate and a recruiter or hiring manager.
Your task is to analyze the transcript, summarize the main points, and extract specific pieces of information about the candidate.

If the information is not present, just mention that the information is not available.

Return a JSON object with the following schema: 
```
{schema}
```
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
Here is the transcript:

<transcript>
{speech_transcribed}
</transcript>

Based on the content of the transcript, please extract the following information about the job candidate:
Carefully review the transcript and extract as much of the requested information as possible.
                    """,
                }
            ],
            model=self.model_name, # "llama3-70b-8192", 
            response_format={"type": "json_object"}
        )

        output = chat_completion.choices[0].message.content

        #process output as dict \
        try:
            output_dict = eval(output)
        except:
            output_dict = json.loads(output)
        output_str = "<json>\n"
        for key, value in output_dict.items():
            output_str += f"{key}: {value} \n"
        output_str += "</json>\n"
        return output_str

    def _chain_question_answering(self, ):
        print('\t _chain_question_answering')
        prompt = PromptTemplate(
            template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI agent that uses contextual information to answer user queries.
I will provide you with some context, followed by a question from the user.
Your task is to use the context to provide a helpful answer to the user's question.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Here is the context you can use to answer the question:
<context>
{context}
</context>

Here is the user's question:
<question>{question}</question>

First, carefully read through the provided context and the user's question.
Think about how you can use the information in the context to address the question.
Write out your thought process and reasoning inside <scratchpad> tags.

After you've thought it through, provide your final answer to the user's question inside <result> tags.
Do not simply copy word-for-word from the context.
Instead, synthesize the relevant information from the context into an answer in your own words.
Keep your answer concise but informative.
If you are unsure or if the question cannot be answered based on the context, say so.

Remember to close the <result> tags with </result> tag.
Otherwise, end the conversation with </result> done.
Finally, say: the above analysis of scratchpad and results is done
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
            """.strip(' \n'),
            input_variables=["speech_transcribed"],
        )

        question_answering_chain = prompt | self.GROQ_LLM | StrOutputParser()
        return question_answering_chain

    def setup_chains(self, ):
        print('\n--- setup_chains')
        self.resume_parser_chain = self._chain_resume_parser
        self.speech_parser_chain = self._chain_speech_parser
        self.question_answering_chain = self._chain_question_answering()
        return

    # ------------------------------------------------------------
    # Nodes
    def _node_get_intent(self, state):
        question = state.get("question", "")
        attachment = state.get("attachment", "")

        if attachment:
            if attachment.endswith('.pdf'):
                intent = "1_upload_resume"
            if attachment.endswith('.wav'):
                intent = "2_upload_speech"
        elif question=='get_summary':
            intent = "3_get_summary"
        elif question=='match_profiles':
            intent = "5_match_profiles"
        else:
            #  intent = self.replan_code_chain.invoke({"question": question,})
            intent = "4_ask_question"

        if self.verbose: print(f"\t intent = {intent}")

        return {
            "question": question,
            "attachment": attachment,
                "intent": intent,
        }

    def _node_resume_parser(self, state):
        resume_path = state['attachment']
        print(f'\tUploaded resume from {resume_path}')
        resume_text, resume_key = self.resume_parser_tool(resume_path)
        input_data = {"resume_text": resume_text}
        resume_content = self.resume_parser_chain(input_data['resume_text'])
        print(f'\tresume_content = ')
        print(resume_content)
        self.resume_full = resume_text

        return {
            "attachment": resume_path,
                "resume_content": resume_content,
                "resume_key": resume_key,
        }

    def _node_speech_parser(self, state):
        speech_path = state['attachment']
        print(f'\tUploaded speech from {speech_path}')

        speech_key, speech_text = self.speech_parser_tool(speech_path)
        print(f'\tspeech_key = {speech_key}')
        speech_transcribed = speech_text['text']
        print(f'\tspeech_transcribed = ')
        print(speech_transcribed)

        input_data = {"speech_transcribed": speech_transcribed}
        speech_keypoints = self.speech_parser_chain(input_data['speech_transcribed'])
        print(f'\tspeech_keypoints = ')
        print(speech_keypoints)
        speech_keypoints = self.process_xml_tags(speech_keypoints, tag='json')
        print(f'\tspeech_keypoints = ')
        print(speech_keypoints)

        return {
            "attachment": speech_path,
                "speech_key": speech_key,
                "speech_transcribed": speech_transcribed,
                "speech_keypoints": speech_keypoints,
        }

    def update_conversation(self, speech_key, speech_transcribed, speech_keypoints):
        data_retrieved = self.supabase_retrieve()

        conversation_key_lst = data_retrieved.get('conversation_key')
        if conversation_key_lst is None:
            conversation_key_lst=[]
        conversation_key_lst.append(speech_key)

        conversation_lst = data_retrieved.get('conversation')
        if conversation_lst is None:
            conversation_lst=[]
        conversation_lst.append(speech_transcribed)

        conversation_keypoints_lst = data_retrieved.get('conversation_keypoints')
        if conversation_keypoints_lst is None:
            conversation_keypoints_lst=[]
        conversation_keypoints_lst.append(speech_keypoints)

        self.supabase_update_column(col_name="conversation_key", col_value=conversation_key_lst)
        self.supabase_update_column(col_name="conversation", col_value=conversation_lst)
        self.supabase_update_column(col_name="conversation_keypoints", col_value=conversation_keypoints_lst)

        # pprint(data_retrieved)
        # pdb.set_trace()
        data_retrieved = self.supabase_retrieve()
        pprint(data_retrieved)
        # pdb.set_trace()
        return
    def _node_store_data(self, state):
        attachment = state.get("attachment", "")

        if attachment.endswith('.pdf'):
            resume_content = state["resume_content"]
            resume_key = state["resume_key"]
            self.RAG_add('resume', [resume_key, resume_content])
            answer = f"Uploaded resume_key={resume_key}"
        elif attachment.endswith('.wav'):
            speech_key = state["speech_key"]
            speech_transcribed = state["speech_transcribed"]
            speech_keypoints = state["speech_keypoints"]
            self.RAG_add('speech', [speech_key, speech_transcribed, speech_keypoints])
            answer = f"Uploaded speech_key={speech_key}"

        return {
            "answer": answer,
        }

    def _node_summary(self, state):
        RAG_context = self.RAG_search(content_level='summary_lite')
        return {
            "answer": RAG_context,
        }

    def _node_RAG_candidate_profile(self, state):
        question = state.get("question", "")
        RAG_context = self.RAG_search(content_level='full')
        return {
            "question": question,
            "answer": RAG_context # proxy
        }

    def _node_qna(self, state):
        question = state.get("question", "")
        RAG_context = state.get("answer", "")

        # Inference to get answer
        input_data = {"question": question,
                      "context": RAG_context,
                     }
        answer_scratchpad_result = self.question_answering_chain.invoke(input_data)

        # Parse answer
        answer = self.process_xml_tags(answer_scratchpad_result, tag='result')

        return {
            "answer": answer,
        }

    def hyde_generate_job_description(self, resume: str) -> str:
        """
        This function uses the Hyde technique where we take a candidates resume and generate a job description for which he will be suitable for.
        """

        return """
Based on the candidate's resume and conversation summaries, I've generated a job description that aligns with their skills, experience, and preferences:

**Job Title:** Sustainability Engineer - Machine Learning Specialist

**Job Summary:**

We are seeking a highly motivated and experienced Sustainability Engineer with a strong background in machine learning to join our team. The ideal candidate will have a passion for environmental sustainability and possess expertise in designing solutions that integrate environmental, human health, and economic considerations. As a machine learning specialist, you will leverage your skills in modeling and drafting software to develop innovative solutions that promote sustainable development and economic growth while adhering to ethical standards.

**Responsibilities:**

* Develop and implement machine learning models to analyze and predict environmental impact, identifying opportunities for sustainable development and cost savings
* Collaborate with cross-functional teams to design and implement sustainable solutions that integrate environmental, human health, and economic considerations
* Utilize expertise in AutoCAD, ALGOR, eQUEST, and EnergyPro to create simulations, models, and prototypes that demonstrate sustainable engineering principles
* Work closely with senior engineers to receive feedback and guidance on project deliverables
* Contribute to the development of new technological stacks and the improvement of existing ones
* Participate in project planning, management, and execution, ensuring timely and efficient delivery of projects
* Stay up-to-date with industry trends, best practices, and emerging technologies in sustainability and machine learning

**Requirements:**

* Bachelor's degree in Engineering Technology, Electrical Energy Engineering, or a related field
* 2+ years of experience in environmental engineering, sustainability, or a related field
* Proficiency in machine learning, modeling, and drafting software such as AutoCAD, ALGOR, eQUEST, and EnergyPro
* Strong understanding of environmental and human health constraints, and their impact on sustainable development and economic growth
* Excellent project management, collaboration, and communication skills
* Ability to work in a fast-paced environment, prioritizing multiple projects and deadlines
* Strong motivation to learn and adapt to new technologies and methodologies

**What We Offer:**

* Competitive salary package (SGD 100,000 per year)
* Opportunities for career growth and professional development in a dynamic and supportive team
* Collaborative and inclusive work culture with regular feedback loops from senior engineers
* 18 days of holiday leave and 30 days of sick leave per year
* Opportunities to work on innovative projects that drive sustainable development and positive environmental impact

If you are a motivated and experienced sustainability engineer with a passion for machine learning, we encourage you to apply for this exciting opportunity.
"""
        chat_completion = self.GROQ_v2.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
The candidate has a background in machine learning.
The candidates resume will be provided.
Please generate a job description for the candidate.
""".strip(' \n')
                },
                {
                    "role": "user",
                    "content": "The candidate resume is: \n\n " + resume
                }
            ],
            model="llama3-70b-8192",
        )

        return chat_completion.choices[0].message.content
    def _node_generate_profile(self, state):

        # Get the candidate profile
        candidate_profile = self.RAG_search(content_level='summary_more')

        # Perform HyDE
        hyde_simulated_job = self.hyde_generate_job_description(candidate_profile)

        return {
            "answer": hyde_simulated_job # proxy for answer
        }

    def _node_RAG_job_portal(self, state):
        hyde_simulated_job = state["answer"]

        # Get embedding
        response = self.OpenAI_LLM.embeddings.create(
            input=hyde_simulated_job,
            model="text-embedding-3-small"
        )
        response = self.OpenAI_LLM.embeddings.create(     input=hyde_simulated_job,     model="text-embedding-3-small" )
        embedding = response.data[0].embedding

        # Get best jobs
        output = self.supabase_client.rpc('match_documents' ,
                                          {"query_embedding" : embedding,
                                           "match_threshold" : 0.5,
                                           "match_count" : 5,
                                          }
                                         ).select('*').execute()
        # output = self.supabase_client.rpc('match_documents' ,    {"query_embedding" : embedding,     "match_threshold" : 0.5,     "match_count" : 5,    }   ).select('*').execute()
        print(f'\tlen(output.data) = {len(output.data)}')
        Recommended_Jobs = ""
        for job_opportunities in output.data:
            job_opportunities_str = job_opportunities['content'].split('**Job Title', 1)[1]
            job_opportunities_str = "**Job Title" + job_opportunities_str
            job_opportunities_str = job_opportunities_str[:100].strip('\n ')
            job_opportunities_str = job_opportunities_str.rstrip('*')
            Recommended_Jobs += f"```\n{job_opportunities_str}\n```\n****************************************************************\n"

        return {
            "answer": Recommended_Jobs
        }

    # ------------------------------------------------------------
    # Edges
    def _edge_routing_to_agents(self, state):
        """
        Route user action to respective agents.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        if self.verbose: print("---[EDGE] _edge_routing_to_agents---")
        intent = state["intent"]
        return intent

    # ------------------------------------------------------------
    # Agents
    def setup_agent(self):
        print('\n--- setup_agent')
        # langgraph.graph
        workflow = StateGraph(GraphState)

        ## Define the nodes
        workflow.add_node("NODE_get_intent", self._node_get_intent)
        workflow.add_node("NODE_resume_parser", self._node_resume_parser)
        workflow.add_node("NODE_speech_parser", self._node_speech_parser)
        workflow.add_node("NODE_store_data", self._node_store_data)
        workflow.add_node("NODE_summary", self._node_summary)
        workflow.add_node("NODE_qna", self._node_qna)
        workflow.add_node("NODE_generate_profile", self._node_generate_profile)
        workflow.add_node("NODE_RAG_job_portal", self._node_RAG_job_portal)
        workflow.add_node("NODE_RAG_candidate_profile", self._node_RAG_candidate_profile)

        ## Add Edges
        workflow.set_entry_point("NODE_get_intent")
        workflow.add_conditional_edges(
            "NODE_get_intent",
            self._edge_routing_to_agents,
            {
                "1_upload_resume": "NODE_resume_parser",
                "2_upload_speech": "NODE_speech_parser",
                "3_get_summary": "NODE_summary",
                "4_ask_question": "NODE_RAG_candidate_profile",
                "5_match_profiles": "NODE_generate_profile",
            },
        )

        workflow.add_edge("NODE_resume_parser", "NODE_store_data")
        workflow.add_edge("NODE_speech_parser", "NODE_store_data")
        workflow.add_edge("NODE_generate_profile", "NODE_RAG_job_portal")
        workflow.add_edge("NODE_RAG_candidate_profile", "NODE_qna")
        ## To sink node
        workflow.add_edge("NODE_store_data", END)
        workflow.add_edge("NODE_summary", END)
        workflow.add_edge("NODE_qna", END)
        workflow.add_edge("NODE_RAG_job_portal", END)

        # Compile
        app = workflow.compile()

        if 1: # Show Image
            app_img = app.get_graph().draw_mermaid_png()
            image = Image.open(io.BytesIO(app_img))
            image.save("graph.png")

        return app

    # ------------------------------------------------------------
    # UI/UX
    def respond(self, input_data, chatbot):
        """
        input_data = {"question": question}
        input_data = {"attachment": 'resume.pdf'}
        input_data = {"attachment": 'Interview.wav'}
        """
        print('\n----------------------------')
        print('----- Start of respond -----')
        print('----------------------------')

        ## invoke the agent (to run it wholesale and get output directly
        question = input_data.get("question", "")
        attachment = input_data.get("attachment", "")

        output = self.app.invoke(input_data, debug=False)
        print(f"output =")
        print(json.dumps(output, indent=2))

        answer = output['answer']
        if answer:
            if attachment:
                question = f"Uploading {attachment}"
            chatbot.append([question, answer])

        print('----------------------------')
        print('------ End of respond ------')
        print('----------------------------\n')
        return "", chatbot
