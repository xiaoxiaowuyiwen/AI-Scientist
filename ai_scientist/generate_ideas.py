import argparse
import json
import os
import os.path as osp
import time
import traceback
from typing import List, Dict, Union

import backoff
import requests

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers
from ai_scientist.logger import debug_logger

S2_API_KEY = os.getenv("S2_API_KEY")

# todo: 这里的area和idea是需要从外部传入的，这里只是一个示例
area = "computer science"
idea = "Use Gen AI to improve the accuracy of recognition models, that is, Gen AI is model A, attack recognition " \
       "model B, so that model B recognition is more accurate. How to deal with Gen AI attacks is to improve " \
       "recognition accuracy, not attack accuracy, which contributes to urban safety."

# 根据area和idea生成一个seed idea
paper_seed_idea_prompt = """
You are a senior doctor in {area}, especially familiar with AI and machine learning. Now I am studying a 
field whose core idea is: "{idea}", Now I need to write a paper based on this idea, but I don't know how to determine 
the topic and description of the paper, as well as the experimental requirements. Please generate such content for me. 
The number of experimental requirements should not exceed 3, and the whole needs to include 3 parts: topic, description,
 and experimental requirements. Please use json format for output, similar to:
```json
{
"Name": "Topic of the paper",
"Title": Full name of the paper,
"Experiment": "Experimental requirements"
}
```
Please note that only json needs to be output, and no other content needs to be output.
"""  # .format(area="computer science", idea="")

idea_first_prompt_bak = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

# 相比原来的idea_first_prompt，去掉了experiment.py的内容
idea_first_prompt = """{task_description}

Here are the ideas that you have already generated:

'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments and directions.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


# GENERATE IDEAS
def generate_ideas(
        base_dir,
        client,
        model,
        skip_generation=False,
        max_num_generations=20,  # 生成的idea的数量
        num_reflections=5,
        log_id=None,
):
    debug_logger.info(
        f'log_id: {log_id}, now in generate_ideas, base_dir: {base_dir}, client: {client}, model: {model}, skip_generation: {skip_generation}, max_num_generations: {max_num_generations}, num_reflections: {num_reflections}')

    if skip_generation:  # 如果跳过idea的生成，则直接从文件中读取已有的ideas
        # Load existing ideas from file
        debug_logger.info(f'log_id: {log_id}, Skipping idea generation and using existing ideas.')
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            debug_logger.info(f"log_id: {log_id}, Loaded existing ideas from file: {base_dir}/ideas.json finished")
            for idea in ideas:
                debug_logger.info(f'log_id: {log_id}, now got one idea: {idea}')
                debug_logger.info(idea)
            return ideas
        except FileNotFoundError:
            debug_logger.info(f"log_id: {log_id}, No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            debug_logger.info(f"log_id: {log_id}, Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []  # 这个是最终的idea的list，包括了seed ideas，以及后续生成的ideas

    # 读取seed ideas
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
        debug_logger.info(f'log_id: {log_id}, load seed ideas from file: {base_dir}/seed_ideas.json finished')

    # 将seed ideas转换为json字符串，加入到idea_str_archive中
    for seed_idea in seed_ideas:
        debug_logger.info(f'log_id: {log_id}, \tgot one seed_idea: {seed_idea}')
        idea_str_archive.append(json.dumps(seed_idea))

    # 读取experiment.py
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
        debug_logger.info(f'log_id: {log_id}, load code from file: {base_dir}/experiment.py finished')

    # 读取prompt.json
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        debug_logger.info(f'log_id: {log_id}, load prompt from file: {base_dir}/prompt.json finished')

    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):  # 迭代max_num_generations次，相当于生成max_num_generations个新idea
        debug_logger.info(f"log_id: {log_id}, Generating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)
            debug_logger.info(f'log_id: {log_id}, prev_ideas_string: {prev_ideas_string}')

            msg_history = []
            debug_logger.info(f"log_id: {log_id}, Iteration 1/{num_reflections}")
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    # task_description=idea_system_prompt,  # 原始的description与llm强相关，此处先直接用system message代替
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            debug_logger.info(f'log_id: {log_id}, idea_first, text: {text}, json_output: {json_output}')

            # Iteratively improve task.
            if num_reflections > 1:  # 如果num_reflections大于1，则进行迭代
                for j in range(num_reflections - 1):
                    debug_logger.info(f"log_id: {log_id}, Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)  # 获取迭代后的idea，这个会覆盖上面的json_output
                    assert (json_output is not None), "Failed to extract JSON from LLM output"
                    debug_logger.info(f'log_id: {log_id}, idea_reflection, text: {text}, json_output: {json_output}')

                    if "I am done" in text:  # 这个I am done是存在于idea_reflection_prompt模板里的
                        debug_logger.info(f"log_id: {log_id}, Idea generation converged after {j + 2} iterations.")
                        break

            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            traceback.print_exc()
            debug_logger.info(f"log_id: {log_id}, Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        debug_logger.info(f'log_id: {log_id}, got a final idea: {idea_str}')
        ideas.append(json.loads(idea_str))
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


# GENERATE IDEAS OPEN-ENDED
def generate_next_idea(
        base_dir,
        client,
        model,
        prev_idea_archive=[],
        num_reflections=5,
        max_attempts=10,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        # seed the archive on the first run with pre-existing ideas
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        # task_description=prompt["task_description"],
                        task_description=idea_system_prompt,  # 原始的description与llm强相关，此处先直接用system message代替
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # Iteratively improve task.
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        text, msg_history = get_response_from_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        ## PARSE OUTPUT
                        json_output = extract_json_between_markers(text)
                        assert (
                                json_output is not None
                        ), "Failed to extract JSON from LLM output"
                        print(json_output)

                        if "I am done" in text:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

    ## SAVE IDEAS
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


'''
@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers
'''


@backoff.on_exception(backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None
    CORE_API_KEY = os.getenv("CORE_API_KEY")
    print(f'in search_for_papers, query: {query}, result_limit: {result_limit}, CORE_API_KEY: {CORE_API_KEY}')
    headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
    payload = {"q": query, "limit": result_limit}
    rsp = requests.post(
        "https://api.core.ac.uk/v3/search/works",
        data=json.dumps(payload),
        headers=headers
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()

    total = results["totalHits"]
    if total == 0:
        return None
    papers = results["results"]
    return papers


novelty_system_msg_bak = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

novelty_system_msg = """You are an ambitious PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
"""

novelty_prompt_bak = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''

novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''


# 这个check_idea_novelty函数是用来检查idea是否novel的，但是并没有修改idea的任何基本信息，只是增加了novel字段
def check_idea_novelty(
        ideas,
        base_dir,
        client,
        model,
        max_num_iterations=10,
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                ## SEARCH FOR PAPERS
                query = json_output["Query"]
                papers = search_for_papers(query, result_limit=10)
                if papers is None:
                    papers_str = "No papers found."

                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        # """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                        """{i}: \nTitle: {title}\nAuthors: {authors}\nVenue: {venue}\nYear: {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            # venue=paper["venue"],
                            venue=paper.get("venue", "Unknown"),
                            # year=paper["year"],
                            year=paper.get("year", "Unknown"),
                            # cites=paper["citationCount"],
                            cites=paper.get("citationCount", "Unknown"),
                            # abstract=paper["abstract"],
                            abstract=paper.get("abstract", "Unknown"),
                        )
                    )
                papers_str = "\n\n".join(paper_strings)
                print(f'now papers_str: {papers_str}')
            except Exception as e:
                traceback.print_exc()
                print(f"line 480 Error: {e}")
                continue

        idea["novel"] = novel
        print(f'idea {idx}, novel: {novel}')

    # Save results to JSON file
    results_file = osp.join(base_dir, "ideas.json")
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 32
    NUM_REFLECTIONS = 5

    parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=[
            "claude-3-5-sonnet-20240620",
            "gpt-4o-2024-05-13",
            "deepseek-coder-v2-0724",
            "llama3.1-405b",
        ],
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and use existing ideas.",
    )
    parser.add_argument(
        "--check-novelty",
        action="store_true",
        help="Check novelty of ideas.",
    )
    args = parser.parse_args()

    # Create client
    if args.model == "claude-3-5-sonnet-20240620":
        import anthropic

        print(f"Using Anthropic API with model {args.model}.")
        client_model = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic()
    elif args.model.startswith("bedrock") and "claude" in args.model:
        import anthropic

        # Expects: bedrock/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock()
    elif args.model.startswith("vertex_ai") and "claude" in args.model:
        import anthropic

        # Expects: vertex_ai/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Vertex AI with model {client_model}.")
        client = anthropic.AnthropicVertex()
    elif args.model == "gpt-4o-2024-05-13":
        import openai

        print(f"Using OpenAI API with model {args.model}.")
        client_model = "gpt-4o-2024-05-13"
        client = openai.OpenAI()
    elif args.model == "deepseek-coder-v2-0724":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "deepseek-coder-v2-0724"
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    elif args.model == "llama3.1-405b":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "meta-llama/llama-3.1-405b-instruct"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Model {args.model} not supported.")

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=MAX_NUM_GENERATIONS,
        num_reflections=NUM_REFLECTIONS,
    )
    if args.check_novelty:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
        )
