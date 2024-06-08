import os
import functools
import traceback
import pdb
import sys
from pprint import pprint
from typing_extensions import TypedDict
from typing import List
import random
import pickle
import re
import base64
import json
import argparse

from io import BytesIO
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import httpx
from dotenv import load_dotenv, find_dotenv

# from utils import MathAgent
from utils import RecruiterAgent

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

# ------------------------------------------------------------
@debug_on_error
def main():
    # Model
    load_dotenv(find_dotenv()) # read local .env file
    assert os.environ["GROQ_API_KEY"]  != ''
    assert os.environ["JIGSAW_API_KEY"]  != ''
    assert os.environ["JIGSAW_PUBLIC_KEY"]  != ''
    assert os.environ["SUPABASE_URL"]  != ''
    assert os.environ["SUPABASE_KEY"]  != ''
    assert os.environ["SUPA_BASE_TABLE"]  != ''

    # Model
    chat_bot = RecruiterAgent()
    chat_bot.verbose = True

    # Gradio Interfaces
    with gr.Blocks() as demo:
        with gr.Row():
            chatbot = gr.Chatbot(height=500) # just to fit the notebook
            input_templates = ["what is the education history ?"]
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(label="State your question | [get_summary] | [match_profiles]",
                                 value='\n\n'.join(input_templates),
                                 max_lines=50, # 20
                                )
                btn = gr.Button("Submit")
                clear = gr.ClearButton(components=[msg, chatbot],
                                    value="Clear console")
            with gr.Column():
                upload_button = gr.File(label="Upload pdf/wav",
                                        type="filepath",
                                        # file_types=["pdf", "wav"],
                                    )
        # Button functions
        submit_kwargs = {
            'inputs': [msg, chatbot],
            'outputs': [msg, chatbot],
        }
        def ask_question(question, chatbot):
            print( f"question = {question}" )
            return chat_bot.respond({"question": question},
                                      chatbot
                                     )
        btn.click(ask_question, **submit_kwargs)
        msg.submit(ask_question, **submit_kwargs) # Press enter to submit

        def upload_file(file, chatbot):
            if file is not None:
                print( f"Uploaded file: {file.name}" )
                return chat_bot.respond({"attachment": file},
                                          chatbot
                                         )
            else:
                print( "No file uploaded" )
                return "", chatbot
        upload_button.change(upload_file,
                             inputs=[upload_button, chatbot],
                             outputs=[msg, chatbot],
                            )

    # Gradio Launch
    demo.launch(server_port=5001,
                share=False,
                show_error=True,
               )

if __name__ == "__main__":
    main()

"""
cd ~/Codes/LLM/DiariserAgent/AI_HACKATHON_SG
conda deactivate; conda activate langchain_groq

clear; python -u demo.py

cp -v *.py backup/sanction_agent

jupyter lab --port 8889

"""