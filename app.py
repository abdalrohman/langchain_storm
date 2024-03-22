import asyncio
import json
from pathlib import Path

import nest_asyncio
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores.sklearn import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
)
from langchain_core.runnables import (
    chain as as_runnable,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from langchain_storm.config import config
from langchain_storm.prompts import *
from langchain_storm.storm_structures import *
from langchain_storm.utilities.env import EnvironmentLoader
from langchain_storm.utilities.helper import print_markdown
from langchain_storm.utilities.log import Log
from langchain_storm.utilities.markdown_converter import (
    FileHandler,
    MarkdownContent,
    MarkdownConverter,
)

# Initialize a logger to record program execution details
log_path = config.log_path
log_level = config.log_level
log = Log(
    log_file_path=Path(log_path) / Path("storm.log"),
    level=log_level,
)
logger = log.get_logger()

# -- Initialize streamlit --
if "topic_history" not in st.session_state:
    st.session_state["topic_history"] = []
if "previous_articles" not in st.session_state:
    st.session_state["previous_articles"] = []

st.set_page_config(
    page_title="Storm",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="🔍",
)

# Load environment variables from a .env
if Path(".env").exists():
    env_loader = EnvironmentLoader(env_file_path=".env")
    env_loader.load_envs()
    st.session_state["using_dot_env"] = "True"
else:
    import os

    user_openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Set this if not set the the value in .env file.",
        key="openai_api_key",  # key is used to store the value in st.session_state
    )
    if user_openai_api_key:
        os.environ["OPENAI_API_KEY"] = user_openai_api_key
    user_firework_api_key = st.sidebar.text_input(
        "Fireworks API Key",
        type="password",
        help="Set this if not set the the value in .env file.",
        key="fireworks_api_key",
    )
    if user_firework_api_key:
        os.environ["FIREWORKS_API_KEY"] = user_firework_api_key
    user_google_api_key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        help="Set this if not set the the value in .env file.",
        key="google_api_key",
    )
    if user_google_api_key:
        os.environ["GOOGLE_API_KEY"] = user_google_api_key

    user_tavily_api_key = st.sidebar.text_input(
        "Tavily API Key",
        type="password",
        help="Set this if not set the the value in .env file.",
        key="tavily_api_key",
    )
    if user_tavily_api_key:
        os.environ["TAVILY_API_KEY"] = user_tavily_api_key

    st.session_state["using_dot_env"] = "False"

# select language model
with st.sidebar:
    llm_choice = st.selectbox(
        "Language Model",
        options=["Fireworks", "OpenAI"],
    )
    st.info(f"Using {llm_choice} as the language model.")

if llm_choice == "Fireworks":
    if st.session_state["using_dot_env"] == "False":
        if not st.session_state["fireworks_api_key"]:
            st.error("Please enter a Fireworks API Key")
            st.stop()
    from langchain_fireworks import ChatFireworks

    fast_llm = ChatFireworks(
        model="accounts/fireworks/models/firefunction-v1",
        max_tokens=32_000,
    )
    long_context_llm = ChatFireworks(
        model="accounts/fireworks/models/firefunction-v1",
        max_tokens=32_000,
    )
elif llm_choice == "OpenAI":
    if st.session_state["using_dot_env"] == "False":
        if not st.session_state["openai_api_key"]:
            st.error("Please enter a OpenAI API Key")
            st.stop()
    from langchain_openai import ChatOpenAI

    fast_llm = ChatOpenAI(model="gpt-3.5-turbo")
    long_context_llm = ChatOpenAI(model="gpt-4-turbo-preview")

# select embeddings
with st.sidebar:
    embeddings_choice = st.selectbox(
        "Embeddings",
        options=[
            "GoogleGenerativeAIEmbeddings",
            "OllamaEmbeddings",
            "OpenAIEmbeddings",
            "HuggingFaceBgeEmbeddings",
        ],
    )
    st.info(f"Using {embeddings_choice} as the embeddings.")

embeddings = None
if embeddings_choice == "GoogleGenerativeAIEmbeddings":
    if st.session_state["using_dot_env"] == "False":
        if not st.session_state["google_api_key"]:
            st.error("Please enter a Google API Key")
            st.stop()
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
elif embeddings_choice == "OllamaEmbeddings":
    from langchain_community.embeddings.ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
elif embeddings_choice == "HuggingFaceBgeEmbeddings":
    from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

    model_name = st.sidebar.text_input("Model name from huggingface eg: (BAAI/bge-m3)")
    if model_name:
        model_kwargs = {"device": "cpu", "trust_remote_code": True}
        encode_kwargs = {
            "normalize_embeddings": True
        }  # set True to compute cosine similarity
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    else:
        st.error("Please enter a model name")
elif embeddings_choice == "OpenAIEmbeddings":
    if st.session_state["using_dot_env"] == "False":
        if not st.session_state["openai_api_key"]:
            st.error("Please enter a Fireworks API Key")
            st.stop()
    from langchain_openai.embeddings import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# define vectorstore
vectorstore = SKLearnVectorStore(embeddings)

# select search engine
with st.sidebar:
    search_engine_choice = st.selectbox(
        "Search Engine",
        options=["DuckDuckGo", "Tavily"],
    )
    max_num_results = st.number_input(
        "Max number of results for search engine",
        value=5,
        step=1,
    )
    st.info(f"Using {search_engine_choice} as the search engine.")

if search_engine_choice == "DuckDuckGo":
    from langchain_community.utilities.duckduckgo_search import (
        DuckDuckGoSearchAPIWrapper,
    )

    @tool
    async def search_engine(query: str):
        """Search engine to the internet."""
        results = DuckDuckGoSearchAPIWrapper(max_results=max_num_results)._ddgs_text(
            query
        )
        return [{"content": r["body"], "url": r["href"]} for r in results]
elif search_engine_choice == "Tavily":
    if st.session_state["using_dot_env"] == "False":
        if not st.session_state["tavily_api_key"]:
            st.error("Please enter a Tavily API Key")
            st.stop()
    from langchain_community.tools.tavily_search import TavilySearchResults

    @tool
    async def search_engine(query: str):
        """Search engine to the internet."""
        results = TavilySearchResults(max_results=max_num_results).invoke(query)
        return [{"content": r["content"], "url": r["url"]} for r in results]


# set hyperparameters
# you can set to restrict the (potentially) infinite research breadth:
with st.sidebar:
    number_of_perspectives = st.number_input(
        "Number of perspectives to survey",
        value=1,
    )
    max_num_turns = st.number_input(
        "Max number of conversation turns in step",
        value=5,
    )


def save_article(markdown_content: str, filename: str):
    filehandler = FileHandler(output_dir="./generated_articles")
    converter = MarkdownConverter(file_handler=filehandler)
    markdown_content = MarkdownContent(content=markdown_content)
    html_content = converter.to_html(markdown_content)

    html_file_path = filehandler.save_file(
        filename=f"{filename}.html", content=html_content
    )
    logger.info(f"Saved article to {html_file_path}")
    pdf_file_path = converter.to_pdf(
        html_content=html_content, pdf_filename=f"{filename}.pdf"
    )
    logger.info(f"Saved article to {pdf_file_path}")


with st.sidebar:
    save_article_state = st.checkbox("Save generated article", value=False)
    if save_article_state:
        filename = st.text_input(
            "Filename",
            value="article",
            help="Write the name of the file to save the article after generated don't specify the extension we save the article as html and pdf",
        )

# -- Initialize containers --
main_container = st.container(border=True)

main_container.header("🔍 Storm", anchor=False)
main_container.markdown(
    """
    [STORM](https://arxiv.org/abs/2402.14207) is a research assistant designed by Shao, et. al that extends the idea of "outline-driven RAG" for richer article generation.

    STORM is designed to generate Wikipedia-style ariticles on a user-provided topic. It applies two main insights to produce more organized and comprehensive articles:

    1. Creating an outline (planning) by querying similar topics helps improve coverage.
    2. Multi-perspective, grounded (in search) conversation simulation helps increase the reference count and information density. 
    """
)

user_topic = main_container.text_input("Topic", placeholder="Enter a topic")

run_button = main_container.button("Generate article", type="primary")

# Apply nest_asyncio to enable nested use of asyncio.run and loop.run_until_complete
nest_asyncio.apply()

# --- Initialize Chains ---
generate_initial_outlines = direct_gen_outline_prompt | fast_llm.with_structured_output(
    Outline
)

expands_generated_outlines = (
    gen_related_topics_prompt | fast_llm.with_structured_output(RelatedSubjects)
)

generate_perspectives = gen_perspectives_prompt | fast_llm.with_structured_output(
    Perspectives
)

gen_queries_chain = gen_queries_prompt | fast_llm.with_structured_output(
    Queries, include_raw=True
)

gen_answer_chain = gen_answer_prompt | fast_llm.with_structured_output(
    AnswerWithCitations, include_raw=True
).with_config(run_name="GenerateAnswer")

refine_outline_chain = refine_outline_prompt | long_context_llm.with_structured_output(
    Outline
)

writer = writer_prompt | long_context_llm | StrOutputParser()

# --- Generate perspectives ---
wikipedia_retriever = WikipediaRetriever(
    load_all_available_meta=True, top_k_results=number_of_perspectives
)


@as_runnable
async def survey_subjects(topic: str):
    related_subjects = await expands_generated_outlines.ainvoke({"topic": topic})
    retrieved_docs = await wikipedia_retriever.abatch(
        related_subjects.topics, return_exceptions=True
    )
    all_docs = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)
    formatted = format_docs(all_docs)
    return await generate_perspectives.ainvoke(
        {"examples": formatted, "topic": topic}
    )  ######


# --- Final Flow ---
@as_runnable
async def generate_question(state: InterviewState):
    editor = state["editor"]
    gn_chain = (
        RunnableLambda(swap_roles).bind(name=editor.name)
        | gen_qn_prompt.partial(persona=editor.persona)
        | fast_llm
        | RunnableLambda(tag_with_name).bind(name=editor.name)
    )
    result = await gn_chain.ainvoke(state)
    return {"messages": [result]}


# --- Answer questions ---
async def gen_answer(
    state: InterviewState,
    config: Optional[RunnableConfig] = None,
    name: str = "Subject Matter Expert",
    max_str_len: int = 15000,
):
    swapped_state = swap_roles(state, name)  # Convert all other AI messages
    queries = await gen_queries_chain.ainvoke(swapped_state)
    query_results = await search_engine.abatch(
        queries["parsed"].queries, config, return_exceptions=True
    )
    successful_results = [
        res for res in query_results if not isinstance(res, Exception)
    ]
    all_query_results = {
        res["url"]: res["content"] for results in successful_results for res in results
    }
    # We could be more precise about handling max token length if we wanted to here
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_call = queries["raw"].additional_kwargs["tool_calls"][0]
    tool_id = tool_call["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    # Only update the shared state with the final answer to avoid
    # polluting the dialogue history with intermediate messages
    generated = await gen_answer_chain.ainvoke(swapped_state)
    cited_urls = set(generated["parsed"].cited_urls)
    # Save the retrieved information to a the shared state for future reference
    cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
    formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
    return {"messages": [formatted_message], "references": cited_references}


# --- Construct the Interview Graph ---
def route_messages(state: InterviewState, name: str = "Subject Matter Expert"):
    messages = state["messages"]
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    if num_responses >= max_num_turns:
        return END
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        return END
    return "ask_question"


builder = StateGraph(InterviewState)

builder.add_node("ask_question", generate_question)
builder.add_node("answer_question", gen_answer)
builder.add_conditional_edges("answer_question", route_messages)
builder.add_edge("ask_question", "answer_question")

builder.set_entry_point("ask_question")
interview_graph = builder.compile().with_config(run_name="Conduct Interviews")


async def initialize_research(state: ResearchState):
    topic = state["topic"]
    coros = (
        generate_initial_outlines.ainvoke({"topic": topic}),
        survey_subjects.ainvoke(topic),
    )
    results = await asyncio.gather(*coros)
    return {
        **state,
        "outline": results[0],
        "editors": results[1].editors,
    }


async def conduct_interviews(state: ResearchState):
    topic = state["topic"]
    initial_states = [
        {
            "editor": editor,
            "messages": [
                AIMessage(
                    content=f"So you said you were writing an article on {topic}?",
                    name="Subject Matter Expert",
                )
            ],
        }
        for editor in state["editors"]
    ]
    # We call in to the sub-graph here to parallelize the interviews
    interview_results = await interview_graph.abatch(initial_states)

    return {
        **state,
        "interview_results": interview_results,
    }


def format_conversation(interview_state):
    messages = interview_state["messages"]
    convo = "\n".join(f"{m.name}: {m.content}" for m in messages)
    return f'Conversation with {interview_state["editor"].name}\n\n' + convo


async def refine_outline(state: ResearchState):
    convos = "\n\n".join(
        [
            format_conversation(interview_state)
            for interview_state in state["interview_results"]
        ]
    )

    updated_outline = await refine_outline_chain.ainvoke(
        {
            "topic": state["topic"],
            "old_outline": state["outline"].as_str,
            "conversations": convos,
        }
    )
    return {**state, "outline": updated_outline}


# TODO handle if vectorstore is empty
async def index_references(state: ResearchState):
    all_docs = []
    for interview_state in state["interview_results"]:
        reference_docs = [
            Document(page_content=v, metadata={"source": k})
            for k, v in interview_state["references"].items()
        ]
        all_docs.extend(reference_docs)
    await vectorstore.aadd_documents(all_docs)
    return state


async def retrieve(
    inputs: dict,
):
    retriever = vectorstore.as_retriever()
    docs = await retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])
    formatted = "\n".join(
        [
            f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )
    return {"docs": formatted, **inputs}


section_writer = (
    retrieve
    | section_writer_prompt
    | long_context_llm.with_structured_output(WikiSection)
)


async def write_sections(state: ResearchState):
    outline = state["outline"]
    sections = await section_writer.abatch(
        [
            {
                "outline": outline,
                "section": section.section_title,
                "topic": state["topic"],
            }
            for section in outline.sections
        ]
    )
    return {
        **state,
        "sections": sections,
    }


async def write_article(state: ResearchState):
    topic = state["topic"]
    sections = state["sections"]
    draft = "\n\n".join([section.as_str for section in sections])
    article = await writer.ainvoke({"topic": topic, "draft": draft})
    return {
        **state,
        "article": article,
    }


builder_of_storm = StateGraph(ResearchState)
nodes = [
    ("init_research", initialize_research),
    ("conduct_interviews", conduct_interviews),
    ("refine_outline", refine_outline),
    ("index_references", index_references),
    ("write_sections", write_sections),
    ("write_article", write_article),
]
for i in range(len(nodes)):
    name, node = nodes[i]
    builder_of_storm.add_node(name, node)
    if i > 0:
        builder_of_storm.add_edge(nodes[i - 1][0], name)

builder_of_storm.set_entry_point(nodes[0][0])
builder_of_storm.set_finish_point(nodes[-1][0])
storm = builder_of_storm.compile()


# --- main ---
async def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = None
        async for step in storm.astream(
            {
                "topic": user_topic,
            },
            # {"recursion_limit": 100},
        ):
            name = next(iter(step))
            logger.info(name)
            logger.info(f"-- {str(step[name])[:300]}")
            if END in step:
                results = step
        if results:
            logger.info(results)
            article = results[END]["article"]

            print_markdown(article.replace("\n#", "\n##"))
        else:
            logger.info("No results")

        return results[END]["article"].replace("\n#", "\n##")
    finally:
        loop.close()


# Run the main coroutine
if run_button:
    with st.spinner("⏳ Generating article..."):
        article = asyncio.run(main())
        if article:
            st.session_state["topic_history"].append(user_topic)
            st.session_state["previous_articles"].append(article)

            with st.container(border=True, height=300):
                st.markdown(article)
                # save the article to the file output folder
                if save_article_state:
                    save_article(
                        markdown_content=article,
                        filename=filename,
                    )

# Display the previous articles and topics in expanded widgets
if st.session_state["previous_articles"] and st.session_state["topic_history"]:
    with main_container.expander("📚 Previous topics"):
        for i, topic in enumerate(st.session_state["topic_history"], start=1):
            st.write(f"{i}- {topic}")
    with st.container(border=True):
        with main_container.expander("📄 Previous articles"):
            for topic_history, previous_articles in zip(
                st.session_state["topic_history"], st.session_state["previous_articles"]
            ):
                st.write(f"Topic: {topic_history}")
                st.markdown(f"{previous_articles}")
                st.markdown("-------------------")
