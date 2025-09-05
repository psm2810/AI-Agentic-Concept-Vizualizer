import streamlit as st
import os
import json
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
# --- PDF Reader Library ---
import pypdf
# --- Secure API Key Input ---
from getpass import getpass
# NO LONGER NEEDED: import httpx
# NO LONGER NEEDED: from openai import OpenAI

# --- Initialization and State Management ---
st.set_page_config(layout="wide", page_title="Ford AI Concepting Suite")

# --- SECURE API KEY HANDLING ---
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API Key: ")
# RE-ADDED: This is required for Gemini
if "TAVILY_API_KEY" not in os.environ:
    os.environ["TAVILY_API_KEY"] = getpass("Enter your Tavily API Key: ")

# --- NEW, SIMPLER PROXY CONFIGURATION ---
# The libraries will automatically use these environment variables.
# IMPORTANT: Replace with your actual proxy URL.
proxy_url = "http://internet.ford.com:83"
os.environ["HTTP_PROXY"] = proxy_url
os.environ["HTTPS_PROXY"] = proxy_url

# --- Initialize models ---
try:
    # The proxy is now handled automatically by the environment variables.
    # No need for httpx or a custom http_client.
    gpt4_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.4, 
        max_tokens=1024
    )
    dalle_client = gpt4_llm.client

    # CORRECTED Gemini model name.
    # It does not need the proxy as it's a Google service.
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.8)

except Exception as e:
    st.error(f"API Key or Model Initialization Error. Please check your keys and model names. Details: {e}")
    st.stop()

# --- Session State (No changes) ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'input'
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Agent and Graph Definitions (No changes needed) ---
class VisualizerState(TypedDict):
    user_inputs: dict; market_analysis: str; new_ideas: List[str]; candidate_prompts: List[str]
    selected_prompts: List[str]; final_concepts: List[dict]; concept_buckets: dict

def clarification_agent(user_prompt: str, prd_text: str):
    st.write("🤖 **Clarification Agent:** Analyzing your request...")
    prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant for vehicle concept design. Based on the user's prompt and PRD, generate 3-4 insightful follow-up questions to clarify their vision.
        **User Prompt:** "{user_prompt}" | **PRD Content:** "{prd_text}"
        Return a JSON object: {{"questions": ["question 1", "question 2"]}}""")
    chain = prompt | gpt4_llm
    response = chain.invoke({"user_prompt": user_prompt, "prd_text": prd_text})
    return json.loads(response.content).get("questions", [])

def market_analysis_agent(state: VisualizerState):
    st.write("📈 **Market Analysis Agent:** Researching...")
    inputs = state['user_inputs']
    prompt = ChatPromptTemplate.from_template(
        """As a Market Analyst, provide a concise analysis based on the user's vision.
        **Initial Prompt:** {prompt} | **PRD Summary:** {prd} | **Clarifications:** {answers}
        Identify: 1. Competitor weaknesses. 2. Market 'white space'. 3. Relevant trends.""")
    chain = prompt | gpt4_llm
    response = chain.invoke({"prompt": inputs['prompt'], "prd": inputs['prd_text'], "answers": inputs['answers']})
    return {"market_analysis": response.content}

def new_ideas_agent(state: VisualizerState):
    st.write("💡 **New Ideas Agent:** Brainstorming...")
    prompt = ChatPromptTemplate.from_template(
        """As a Creative Strategist, generate 3 DIVERGENT vehicle concepts based on the market analysis.
        **Market Analysis:** {market_analysis}
        Return a JSON object: {{"ideas": ["idea 1", "idea 2", "idea 3"]}}""")
    chain = prompt | gpt4_llm
    response = chain.invoke({"market_analysis": state["market_analysis"]})
    return {"new_ideas": json.loads(response.content).get("ideas", [])}

def prompt_generator_agent(state: VisualizerState):
    st.write("✍️ **Prompt Generator Agent:** Crafting prompts...")
    all_prompts = []; template = """Expand this idea into a detailed, photorealistic image prompt: **Core Idea:** {idea}"""; prompt = ChatPromptTemplate.from_template(template)
    gpt4_chain = prompt | gpt4_llm; gemini_chain = prompt | gemini_llm
    for idea in state['new_ideas']:
        all_prompts.append(gpt4_chain.invoke({"idea": idea}).content)
        all_prompts.append(gemini_chain.invoke({"idea": idea}).content)
    return {"candidate_prompts": all_prompts}

def prompt_critic_agent(state: VisualizerState):
    st.write("🧐 **Prompt Critic Agent (GPT-4o):** Selecting for divergence...")
    prompt = ChatPromptTemplate.from_template(
        """As a Creative Director, select the best {num_selected} prompts from the list below for **maximum creative divergence**.
        **Candidate Prompts:** {prompts}
        Return a JSON object: {{"selected": ["prompt 1", "prompt 2"]}}""")
    chain = prompt | gpt4_llm
    response = chain.invoke({"num_selected": state["user_inputs"]["num_images"], "prompts": "\n\n---\n\n".join(state["candidate_prompts"])})
    return {"selected_prompts": json.loads(response.content).get("selected", [])}

def image_generation_node(state: VisualizerState):
    st.write("🎨 **Image Generation Node:** Creating images...")
    final_concepts = []
    for i, p in enumerate(state['selected_prompts']):
        with st.spinner(f"Generating image {i+1}/{len(state['selected_prompts'])}..."):
            res = dalle_client.images.generate(model="dall-e-3", prompt=p, size="1792x1024", n=1)
            final_concepts.append({"prompt": p, "url": res.data[0].url, "id": i})
    return {"final_concepts": final_concepts}

def image_critic_agent(prompt: str):
    st.write("🔄 **Image Critic Agent:** Refining prompt...")
    critic_prompt = ChatPromptTemplate.from_template(
        """As an Art Director, slightly modify this prompt to better capture its spirit.
        **Original:** {prompt} | **Modified:**""")
    chain = critic_prompt | gpt4_llm
    return chain.invoke({"prompt": prompt}).content

def concept_bucketing_agent(state: VisualizerState):
    st.write("🗂️ **Concept Bucketing Agent:** Categorizing results...")
    prompts = [item['prompt'] for item in state['final_concepts']]
    prompt = ChatPromptTemplate.from_template(
        """Group these prompts into 3-4 thematic buckets.
        **Prompts:** {prompts}
        Return JSON: {{"Bucket Name": ["prompt 1"]}}""")
    chain = prompt | gpt4_llm
    response = chain.invoke({"prompts": "\n".join(prompts)})
    return {"concept_buckets": json.loads(response.content)}

# --- Graph Definition (No changes) ---
graph = StateGraph(VisualizerState)
graph.add_node("market_analysis", market_analysis_agent); graph.add_node("new_ideas", new_ideas_agent)
graph.add_node("prompt_generator", prompt_generator_agent); graph.add_node("prompt_critic", prompt_critic_agent)
graph.add_node("generate_images", image_generation_node); graph.add_node("bucket_concepts", concept_bucketing_agent)
graph.set_entry_point("market_analysis"); graph.add_edge("market_analysis", "new_ideas")
graph.add_edge("new_ideas", "prompt_generator"); graph.add_edge("prompt_generator", "prompt_critic")
graph.add_edge("prompt_critic", "generate_images"); graph.add_edge("generate_images", "bucket_concepts")
graph.add_edge("bucket_concepts", END); visualizer_app = graph.compile()

# --- Streamlit UI (No changes needed in logic) ---
st.title("Product Concept Visualizer")

if st.session_state.stage == 'input':
    st.header("Step 1: Define Your Concept")
    with st.form("input_form"):
        uploaded_file = st.file_uploader("Upload Product Requirement Document (Optional)", type=['pdf', 'txt', 'md'])
        view_type = st.selectbox("Select view type for image generation:", ["Exterior", "Interior", "Both"])
        prompt = st.text_area("Enter your product description or prompt:", "A new electric pickup truck with Nostalgic Design Elements from the F100.", height=100)
        num_images = st.number_input("Number of image prompts to generate:", min_value=2, max_value=5, value=3)
        submitted = st.form_submit_button("Analyze Files")
        if submitted:
            prd_text = ""
            if uploaded_file:
                st.info(f"Reading uploaded file: {uploaded_file.name}")
                if uploaded_file.type == "application/pdf":
                    try:
                        pdf_reader = pypdf.PdfReader(uploaded_file)
                        prd_text = "".join(page.extract_text() for page in pdf_reader.pages)
                        st.success("Successfully extracted text from PDF.")
                    except Exception as e:
                        st.error(f"Error reading PDF file: {e}")
                else:
                    file_bytes = uploaded_file.read()
                    try:
                        prd_text = file_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        prd_text = file_bytes.decode("latin-1", errors='ignore')
            st.session_state.user_inputs = {"prompt": prompt, "prd_text": prd_text, "view_type": view_type, "num_images": num_images}
            with st.spinner("Generating follow-up questions..."):
                st.session_state.questions = clarification_agent(prompt, prd_text)
            st.session_state.stage = 'questions'; st.rerun()

if st.session_state.stage == 'questions':
    st.header("Step 2: Follow-up Questions")
    with st.form("questions_form"):
        for i, q in enumerate(st.session_state.questions):
            st.session_state.answers[q] = st.text_area(q, key=f"q_{i}")
        c1, c2 = st.columns(2)
        if c1.form_submit_button("⬅️ Back"): st.session_state.stage = 'input'; st.rerun()
        if c2.form_submit_button("✅ Generate", type="primary"):
            st.session_state.user_inputs['answers'] = st.session_state.answers
            st.session_state.stage = 'processing'; st.rerun()

if st.session_state.stage == 'processing':
    st.header("Generating Concepts...")
    with st.status("Running AI Concepting Workflow...", expanded=True):
        results = visualizer_app.invoke({"user_inputs": st.session_state.user_inputs})
        st.session_state.results = results
        status.update(label="Workflow Complete!", state="complete")
    st.session_state.stage = 'results'; st.rerun()

if st.session_state.stage == 'results':
    st.header("Generated Visual Representations")
    results = st.session_state.results
    if results and 'final_concepts' in results:
        cols = st.columns(len(results['final_concepts']))
        for i, concept in enumerate(results['final_concepts']):
            with cols[i]:
                st.image(concept['url'], use_column_width=True)
                with st.expander("Prompt"): st.write(concept['prompt'])
                if st.button("🔄 Regenerate", key=f"regen_{concept['id']}"):
                    with st.spinner("Image Critic is working..."):
                        new_prompt = image_critic_agent(concept['prompt'])
                        res = dalle_client.images.generate(model="dall-e-3", prompt=new_prompt, size="1792x1024", n=1)
                        st.session_state.results['final_concepts'][i]['url'] = res.data[0].url
                        st.session_state.results['final_concepts'][i]['prompt'] = new_prompt
                        st.rerun()
        st.header("Concept Bucketing")
        if 'concept_buckets' in results and results['concept_buckets']:
            for bucket, prompts in results['concept_buckets'].items():
                with st.container(border=True):
                    st.subheader(f"Bucket: {bucket}")
                    for p in prompts: st.caption(f"• {p}")
    else: st.error("Generation failed.")
    if st.button("⬅️ Start Over"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
