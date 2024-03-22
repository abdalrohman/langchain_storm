import asyncio
import json
import logging
import os
from pathlib import Path

import nest_asyncio
import typer
from langchain_community.retrievers import WikipediaRetriever
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

storm_cli = typer.Typer()

supported_embeddings = [
    "Google",
    "Ollama",
    "OpenAI",
    "HuggingFace",
]
supported_llm = ["Fireworks", "OpenAI"]
supported_search_engine = ["DuckDuckGo", "Tavily"]
default_aritcle_filename = "article"

# Initialize a logger to record program execution details
log_path = config.log_path
log_level = config.log_level
log = Log(
    log_file_path=Path(log_path) / Path("storm_cli.log"),
    level=log_level,
)
log.get_logger()

logger = logging.getLogger("storm.cli")


@storm_cli.command()
def storm(
    *,
    openai_api_key: Optional[str] = typer.Option(
        None,
        "--openai_api_key",
        help="OpenAI API key. Use this if the key is not set in the .env file.",
    ),
    fireworks_api_key: Optional[str] = typer.Option(
        None,
        "--fireworks_api_key",
        help="Fireworks API key. Use this if the key is not set in the .env file.",
    ),
    google_api_key: Optional[str] = typer.Option(
        None,
        "--google_api_key",
        help="Google API key. Use this if the key is not set in the .env file.",
    ),
    tavily_api_key: Optional[str] = typer.Option(
        None,
        "--tavily_api_key",
        help="Tavily API key. Use this if the key is not set in the .env file.",
    ),
    llm_choice: Optional[str] = typer.Option(
        "Fireworks",
        "--llm",
        help=f"Choice of language model {supported_llm}.",
        case_sensitive=False,
        show_choices=True,
    ),
    embeddings_choice: Optional[str] = typer.Option(
        "Google", "--embeddings", help=f"Choice of embeddings. {supported_embeddings}."
    ),
    huggingface_model_name: Optional[str] = typer.Option(
        "BAAI/bge-m3",
        "--huggingface_model",
        help="HuggingFace model name. Use this if you choose HuggingFace as the embeddings.",
    ),
    search_engine_choice: Optional[str] = typer.Option(
        "DuckDuckGo",
        "--search_engine",
        help=f"Choice of search engine. {supported_search_engine}.",
    ),
    max_num_results: Optional[int] = typer.Option(
        5,
        "--max_num_results",
        help="Maximum number of results to return from the search engine.",
    ),
    number_of_perspectives: Optional[int] = typer.Option(
        1, "--number_of_perspectives", help="Number of perspectives to survey."
    ),
    max_num_turns: Optional[int] = typer.Option(
        5, "--max_num_turns", help="Maximum number of conversation turns in each step."
    ),
    save_article_state: Optional[bool] = typer.Option(
        True,
        "--save/--no-save",
        help="Flag to save the generated article. If set, the article will be saved.",
    ),
    filename: Optional[str] = typer.Option(
        default_aritcle_filename,
        "--filename",
        help="Filename to save the generated article. the article will be saved as both HTML and PDF.",
    ),
    user_topic: str = typer.Option(
        ..., "--topic", help="Topic for the article. This is a required option."
    ),
):
    logger.info("Starting Storm...")
    # Load environment variables from a .env
    if Path(".env").exists():
        env_loader = EnvironmentLoader(env_file_path=".env")
        env_loader.load_envs()
    elif openai_api_key or fireworks_api_key or google_api_key or tavily_api_key:
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if fireworks_api_key:
            os.environ["FIREWORKS_API_KEY"] = fireworks_api_key
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
    else:
        logger.error(
            "Please provide required API Keys or set them in .env file to continue"
        )
        raise typer.Abort()

    # setup LLMs
    if llm_choice.lower() == "Fireworks".lower():
        from langchain_fireworks import ChatFireworks

        logger.info("Using Fireworks as the language model.")
        fast_llm = ChatFireworks(
            model="accounts/fireworks/models/firefunction-v1",
            max_tokens=32_000,
        )
        long_context_llm = ChatFireworks(
            model="accounts/fireworks/models/firefunction-v1",
            max_tokens=32_000,
        )
    elif llm_choice.lower() == "OpenAI".lower():
        logger.info("Using OpenAI as the language model.")
        from langchain_openai import ChatOpenAI

        fast_llm = ChatOpenAI(model="gpt-3.5-turbo")
        long_context_llm = ChatOpenAI(model="gpt-4-turbo-preview")
    else:
        logger.error(
            f"Unsupported LLMs choice: {llm_choice}. Supported options are: {supported_llm}"
        )
        raise typer.Abort()

    # setup embeddings
    if embeddings_choice.lower() == "Google".lower():
        logger.info("Using Google as the embeddings.")
        from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif embeddings_choice.lower() == "Ollama".lower():
        logger.info("Using Ollama as the embeddings.")
        from langchain_community.embeddings.ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    elif embeddings_choice.lower() == "OpenAI".lower():
        logger.info("Using OpenAI as the embeddings.")
        from langchain_openai.embeddings import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    elif embeddings_choice.lower() == "HuggingFace".lower():
        logger.info("Using HuggingFace as the embeddings.")
        from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

        if huggingface_model_name:
            model_kwargs = {"device": "cpu", "trust_remote_code": True}
            encode_kwargs = {
                "normalize_embeddings": True
            }  # set True to compute cosine similarity
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=huggingface_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        else:
            logger.error(
                "Please specify --huggingface_model_name and set the huggingface model"
            )
            raise typer.Abort()
    else:
        logger.error(
            f"Unsupported embeddings choice: {embeddings_choice}. Supported options are: {supported_embeddings}"
        )
        raise typer.Abort()

    # setup search engine
    if search_engine_choice.lower() == "DuckDuckGo".lower():
        logger.info("Using DuckDuckGo as the search engine.")
        from langchain_community.utilities.duckduckgo_search import (
            DuckDuckGoSearchAPIWrapper,
        )

        @tool
        async def search_engine(query: str):
            """Search engine to the internet."""
            results = DuckDuckGoSearchAPIWrapper(
                max_results=max_num_results
            )._ddgs_text(query)
            return [{"content": r["body"], "url": r["href"]} for r in results]
    elif search_engine_choice.lower() == "Tavily".lower():
        if not os.environ.get("TAVILY_API_KEY") or tavily_api_key:
            logger.info("Using Tavily as the search engine.")
            from langchain_community.tools.tavily_search import TavilySearchResults

            @tool
            async def search_engine(query: str):
                """Search engine to the internet."""
                results = TavilySearchResults(max_results=max_num_results).invoke(query)
                return [{"content": r["content"], "url": r["url"]} for r in results]
        else:
            logger.error(
                "Please provide a Tavily API Key or set it in .env file to continue"
            )
            raise typer.Abort()
    else:
        logger.error(
            f"Unsupported search engine choice: {search_engine_choice}. Supported options are: {supported_search_engine}"
        )
        raise typer.Abort()

    # hyperparameter
    logger.info(f"Number of perspectives: {number_of_perspectives}")
    logger.info(f"Maximum number of conversation turns: {max_num_turns}")
    logger.info(f"Maximum number of search results: {max_num_results}")

    # handle article save
    if save_article_state:
        logger.info("Saving the generated article is [ON]...")
        from langchain_storm.utilities.markdown_converter import (
            FileHandler,
            MarkdownContent,
            MarkdownConverter,
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

        if filename == default_aritcle_filename:
            logger.warning(
                f"Set --filename to specify the file name for saving article or Otherwise using the default [{default_aritcle_filename}]."
            )
    else:
        logger.info("Saving the generated article is [OFF]...")

    logger.info(f"Topic: [{user_topic}]")

    # define vectorstore
    from langchain_community.vectorstores.sklearn import SKLearnVectorStore

    vectorstore = SKLearnVectorStore(embeddings)

    # Apply nest_asyncio to enable nested use of asyncio.run and loop.run_until_complete
    nest_asyncio.apply()

    # --- Initialize Chains ---
    generate_initial_outlines = (
        direct_gen_outline_prompt | fast_llm.with_structured_output(Outline)
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

    refine_outline_chain = (
        refine_outline_prompt | long_context_llm.with_structured_output(Outline)
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
            res["url"]: res["content"]
            for results in successful_results
            for res in results
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
        cited_references = {
            k: v for k, v in all_query_results.items() if k in cited_urls
        }
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

    article = asyncio.run(main())
    if article:
        if save_article_state:
            save_article(
                markdown_content=article,
                filename=filename,
            )


if __name__ == "__main__":
    storm_cli()
