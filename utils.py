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

# ------------------------------------------------------------
def debug_on_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            pdb.post_mortem(exc_traceback)
    return wrapper

class RecruiterAgent():
    def __init__(self) -> None:
        self.verbose = True
        self.dummy_data = {}
        self.state = {}

        self.GROQ_LLM = self.load_llm_model()
        self.JIGSAW_API_KEY = os.environ["JIGSAW_API_KEY"]
        self.JIGSAW_PUBLIC_KEY = os.environ["JIGSAW_PUBLIC_KEY"]
    
        self.setup_tools()
        self.setup_chains()

        self.verbose = False

        self.warmup()

    def warmup(self, ):
        self.app = self.setup_agent()
        self.app.invoke({"attachment": 'data/gettysburg.wav'})
        self.app.invoke({"attachment": 'data/JohnDoeResume.pdf'})

    # ------------------------------------------------------------
    # Model
    def load_llm_model(self, ):
        print('\n--- load_llm_model')
        model_gemma = "gemma-7b-lt"
        model_llama3_70b = "llama3-70b-8192"
        model_llama3_8b = "llama3-8b-8192"
        model_mixtral = "mixtral-8x7b-32768"

        GROQ_LLM = ChatGroq(model=model_llama3_8b,)
        print(f'\tUsing {GROQ_LLM.model_name}')
        return GROQ_LLM

    # ------------------------------------------------------------
    # Tools
    def _tool_REPL(self):
        from langchain_core.tools import Tool
        from langchain_experimental.utilities import PythonREPL
        return PythonREPL()

    def JIGSAW_upload_resume_file(self, resume_file):
        resume_key = os.path.basename(resume_file).replace(' ', '_')
        assert resume_key.endswith('.pdf')
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

    def JIGSAW_upload_speech_file(self, speech_file):
        speech_key = os.path.basename(speech_file).replace(' ', '_')
        assert speech_key.endswith('.wav')
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
        }
        print(f"\tbody = {body}")

        # Make the POST request
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
        speech_key, data_upload = self.JIGSAW_upload_speech_file(speech_file)

        # transcribe
        speech_transcribed = self.JIGSAW_transcribe_file(speech_key)
        return speech_transcribed

    def setup_tools(self):
        self.python_repl_tool = self._tool_REPL()
        return
        self.python_repl.run("print(1+1)")

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
    def _chain_resume_parser(self, ):
        print('\t _chain_resume_parser')
        prompt = PromptTemplate(
            template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a smart resume summary agent assistant

<|eot_id|>
<|start_header_id|>user<|end_header_id|>
You will be given a resume and you need to split it into different sections like education, experience, skills, etc.
The resume is given below:\n\n{resume_text} 

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
            """.strip(' \n'),
            input_variables=["resume_text"],
        )

        resume_parser_chain = prompt | self.GROQ_LLM | StrOutputParser()
        if 0: # inference first pass
            question = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
            expected_answer = 18
            inputs = {"question": question, "num_steps": 0}
            answer_json = self.invoke_with_retry(resume_parser_chain, inputs)
            print(f"answer_json = {answer_json}")
            print(f"expected_answer = {expected_answer}")
            pdb.set_trace()
        return resume_parser_chain

    def _chain_speech_parser(self, ):
        print('\t _chain_speech_parser')
        prompt = PromptTemplate(
            template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant that specializes in summarizing conversations and extracting key information.
You will be provided with the transcript of an audio conversation between a job candidate and a recruiter or hiring manager.
Your task is to analyze the transcript, summarize the main points, and extract specific pieces of information about the candidate.

<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is the transcript:

<transcript>
{speech_transcribed}
</transcript>

Based on the content of the transcript, please extract the following information about the job candidate:

- Name: The candidate's name 
- Gender: The candidate's gender (either 'male' or 'female' only)
- Languages: A list of the languages the candidate speaks
- Country: The country the candidate is currently residing in
- Work Pass Type: The type of work pass the candidate has (e.g. visa, permanent residency)  
- Nationality: The candidate's nationality
- Notice Period: The amount of notice the candidate needs to give their current employer
- Reason for looking out: Why the candidate is seeking a new job
- Looking for in next move: What the candidate is seeking in their next job/company
- Other Interviews in pipeline: Other companies the candidate is interviewing with
- Current Salary: The candidate's current salary 
- Expected Salary: The salary the candidate is seeking
- Identity: The candidate's NRIC, FIN, or Passport number (for permanent roles only)
- Open to Consulting model: Whether the candidate is open to a consulting arrangement (for contract roles only)
- Current Annual Leave days: The candidate's current number of annual leave days (for contract roles only)
- Worked for Client Before or no: Whether the candidate has worked for the client company before (for contract roles only)
- Hiring for other roles?: Whether the candidate is also hiring for other roles at their current company
- General Notes: Any other relevant information about the candidate

If any of the above information is not available in the transcript, set the value to null.

Please provide your output in valid JSON format, like this:

<json>
{{
  "Name": "John Smith",
  "Gender": "male",
  "Languages": ["English", "Spanish"],
  "Country": "United States",
  "Work Pass Type": null,
  "Nationality": "American",
  "Notice Period": "2 weeks",
  "Reason for looking out": "Seeking new challenges and growth opportunities",
  "Looking for in next move": "A dynamic, fast-paced work environment",
  "Other Interviews in pipeline": ["ABC Corp", "XYZ Inc"],
  "Current Salary": null, 
  "Expected Salary": "$80,000",
  "Identity": null,
  "Open to Consulting model": null,
  "Current Annual Leave days": null,
  "Worked for Client Before or no": null,
  "Hiring for other roles?": "Yes, also hiring for sales and marketing roles",
  "General Notes": "Very enthusiastic and driven candidate"
}}
</json>

Carefully review the transcript and extract as much of the requested information as possible.
Remember, if a piece of information is not available, set the value to null.
Provide your final output in valid JSON format.

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
            """.strip(' \n'),
            input_variables=["speech_transcribed"],
        )

        speech_parser_chain = prompt | self.GROQ_LLM | StrOutputParser()
        if 0: # inference first pass
            question = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
            expected_answer = 18
            inputs = {"question": question, "num_steps": 0}
            answer_json = self.invoke_with_retry(speech_parser_chain, inputs)
            print(f"answer_json = {answer_json}")
            print(f"expected_answer = {expected_answer}")
            pdb.set_trace()
        return speech_parser_chain

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
        self.resume_parser_chain = self._chain_resume_parser()
        self.speech_parser_chain = self._chain_speech_parser()
        self.question_answering_chain = self._chain_question_answering()
        # self. xxx _chain = self._chain_ xxx ()
        # self. xxx _chain = self._chain_ xxx ()
        return

    # ------------------------------------------------------------
    # Nodes
    def _node_get_intent(self, state):
        question = state.get("question", "")
        attachment = state.get("attachment", "")

        if attachment:
            if attachment.endswith('.pdf'):
                intent = "upload_resume"
            if attachment.endswith('.wav'):
                intent = "upload_speech"
        elif question=='get_summary':
            intent = "get_summary"
        else:
            # intent = self.replan_code_chain.invoke({"question": question,})
            intent = "ask_question"

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
        resume_content = self.resume_parser_chain.invoke(input_data)
        print(f'\tresume_content = ')
        print(resume_content)
        self.state['resume_content'] = resume_content
        return {
            "attachment": resume_path,
                "resume_content": resume_content,
                "resume_key": resume_key,
        }

    def _node_speech_parser(self, state):
        speech_path = state['attachment']
        print(f'\tUploaded speech from {speech_path}')

        speech_text = self.speech_parser_tool(speech_path)
        speech_transcribed = speech_text['text']
        print(f'\tspeech_transcribed = ')
        print(speech_transcribed)

        input_data = {"speech_transcribed": speech_transcribed}
        speech_keypoints = self.speech_parser_chain.invoke(input_data)
        print(f'\tspeech_keypoints = ')
        print(speech_keypoints)
        speech_keypoints = self.process_xml_tags(speech_keypoints, tag='json')
        print(f'\tspeech_keypoints = ')
        print(speech_keypoints)
        self.state['speech_keypoints'] = speech_keypoints
        return {
            "attachment": speech_path,
                "speech_transcribed": speech_transcribed,
                "speech_keypoints": speech_keypoints,
        }

    def _node_store_data(self, state):
        attachment = state.get("attachment", "")

        if attachment:
            if attachment.endswith('.pdf'):
                resume_content = state.get("resume_content", "")
                self.dummy_data['resume'] = resume_content
            if attachment.endswith('.wav'):
                speech_transcribed = state.get("speech_transcribed", "")
                speech_keypoints = state.get("speech_keypoints", "")
                self.dummy_data['speech_transcribed'] = speech_transcribed
                self.dummy_data['speech_keypoints'] = speech_keypoints

        # store data into database here for RAG | supabase

        return {
            "answer": "",
        }

    def _node_summary(self, state):
        # retrieve resume / speech_keypoints | from supabase
        print(f'self.state = {self.state}')
        #####
        resume_content = self.state.get("resume_content", "")
        if resume_content is None:
            resume_content = ""
        #####
        speech_keypoints = self.state.get("speech_keypoints", "")
        if speech_keypoints is None:
            speech_keypoints = ""
        #####
        RAG_context = resume_content + "\n" + speech_keypoints

        return {
            "answer": RAG_context,
        }

    def _node_qna(self, state):
        question = state.get("question", "")

        # RAG retrieve relevant question | from supabase
        print(f'self.state = {self.state}')
        #####
        resume_content = self.state.get("resume_content", "")
        if resume_content is None:
            resume_content = ""
        #####
        speech_keypoints = self.state.get("speech_keypoints", "")
        if speech_keypoints is None:
            speech_keypoints = ""
        #####
        RAG_context = resume_content + "\n" + speech_keypoints

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

        ## Add Edges
        workflow.set_entry_point("NODE_get_intent")
        workflow.add_conditional_edges(
            "NODE_get_intent",
            self._edge_routing_to_agents,
            {
                "upload_resume": "NODE_resume_parser",
                "upload_speech": "NODE_speech_parser",
                "get_summary": "NODE_summary",
                "ask_question": "NODE_qna",
            },
        )

        workflow.add_edge("NODE_resume_parser", "NODE_store_data")
        workflow.add_edge("NODE_speech_parser", "NODE_store_data")
        ## To sink node
        workflow.add_edge("NODE_store_data", END)
        workflow.add_edge("NODE_summary", END)
        workflow.add_edge("NODE_qna", END)

        # Compile
        app = workflow.compile()

        # Show Image
        app_img = app.get_graph().draw_mermaid_png()
        image = Image.open(io.BytesIO(app_img))
        image.save("graph.png")

        return app

    # ------------------------------------------------------------
    # UI/UX
    @debug_on_error
    def respond(self, input_data, chatbot):
        # input_data = {"question": question}
        # input_data = {"attachment": 'resume.pdf'}
        # input_data = {"attachment": 'call.wav'}
        print('\n----------------------------')
        print('----- Start of respond -----')
        print('----------------------------')

        ## invoke the agent (to run it wholesale and get output directly
        question = input_data.get("question", "")
        attachment = input_data.get("attachment", "")
        output = self.app.invoke(input_data, debug=False)
        print(f"output ="); print(json.dumps(output, indent=2))
        answer = output['answer']
        if answer:
            chatbot.append([question, answer])

        print('----------------------------')
        print('------ End of respond ------')
        print('----------------------------\n')
        return "", chatbot
