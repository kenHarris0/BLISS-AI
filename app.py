
import os
from apikey import apikey
import platform
import webbrowser
import streamlit as sl
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper
os.environ['OpenAI_API_KEY'] = apikey

# framework
sl.title(' ⚡ BLISS AI')
prompt = sl.text_input('ENTER YOUR PROMPT TO BLISS AI')

title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me about {topic}'
)
script_template = PromptTemplate(
    input_variables=['title', 'wikipediar'],
    template='write me a detailed script based on this title TITLE: {title} while leveraging with this wikipedia reasearch:{wikipediar}'
)
# memort
title_memory = ConversationBufferMemory(
    input_key='topic', memory_key='chat history')
script_memory = ConversationBufferMemory(
    input_key='title', memory_key='chat history')
# instance of openai
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template,
                       verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template,
                        verbose=True, output_key='script', memory=script_memory)
pedia = WikipediaAPIWrapper()
# what my prompt returns
if prompt:
    title = title_chain.run(prompt)
    wikipedias = pedia.run(prompt)
    script = script_chain.run(title=title, wikipediar=wikipedias)

    sl.write(title)
    sl.write(script)

    with sl.expander('title history'):
        sl.info(title_memory.buffer)

    with sl.expander('script history'):
        sl.info(script_memory.buffer)

    with sl.expander('wiki history'):
        sl.info(wikipedias)
