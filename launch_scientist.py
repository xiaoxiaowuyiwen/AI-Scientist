import argparse
import json
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import time
import uuid
from datetime import datetime

import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.logger import debug_logger
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement
from ai_scientist.perform_writeup import perform_writeup, generate_latex

NUM_REFLECTIONS = 3


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
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
        default="claude-3-5-sonnet-20240620",
        choices=[
            "claude-3-5-sonnet-20240620",
            "gpt-4o-2024-05-13",
            "deepseek-coder-v2-0724",
            "llama3.1-405b",
            # Anthropic Claude models via Amazon Bedrock
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/anthropic.claude-3-opus-20240229-v1:0"
            # Anthropic Claude models Vertex AI
            "vertex_ai/claude-3-opus@20240229",
            "vertex_ai/claude-3-5-sonnet@20240620",
            "vertex_ai/claude-3-sonnet@20240229",
            "vertex_ai/claude-3-haiku@20240307",
        ],
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="latex",
        choices=["latex"],
        help="What format to use for writeup",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=50,
        help="Number of ideas to generate",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def worker(
        queue,
        base_dir,
        results_dir,
        model,
        client,
        client_model,
        writeup,
        improvement,
        gpu_id,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            model,
            client,
            client_model,
            writeup,
            improvement,
            log_file=True,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    print(f"Worker {gpu_id} finished.")


def do_idea(
        base_dir,
        results_dir,
        idea,
        model,
        client,
        client_model,
        writeup,
        improvement,
        log_file=False,
):
    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)  # 注意了，这个目录是在results目录下的！！！
    debug_logger.info(f'do_idea, idea_name: {idea_name}, folder_name: {folder_name}')
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)

    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    debug_logger.info(f'do_idea, idea_name: {idea_name}, baseline_results: {baseline_results}')

    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    debug_logger.info(f'do_idea, idea_name: {idea_name}, exp_file: {exp_file}, vis_file: {vis_file}, notes: {notes}')

    # 向notes.txt中写入实验的标题和描述
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")

    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log

    try:
        print_time()
        debug_logger.info(f"*Starting idea: {idea_name}*")

        ## 1. PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]  # 分别是experiment.py, plot.py, notes.txt
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        print_time()
        debug_logger.info(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
        except Exception as e:
            debug_logger.error(f"perform_experiments failed for idea {idea_name}, error: {e}")
            return False

        if not success:
            debug_logger.error(f"perform_experiments failed for idea {idea_name}")
            return False

        print_time()
        debug_logger.info(f"*Starting Writeup*")
        ## 2. PERFORM WRITEUP
        if writeup == "latex":
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            fnames = [exp_file, writeup_file, notes]  # 分别是experiment.py, template.tex, notes.txt
            if model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                main_model = Model(model)
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                perform_writeup(idea, folder_name, coder, client, client_model)
            except Exception as e:
                debug_logger.info(f"Failed to perform writeup: {e}")
                return False
            debug_logger.info("Done writeup")
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        print_time()
        debug_logger.info(f"*Starting Review*")
        ## 3. REVIEW PAPER
        if writeup == "latex":
            try:
                paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
                review = perform_review(
                    paper_text,
                    # model="gpt-4o-2024-05-13",
                    model=model,
                    # client=openai.OpenAI(), # 这里怎么写死了？S
                    client=client,
                    # num_reflections=5,
                    num_reflections=1,
                    num_fs_examples=1,
                    # num_reviews_ensemble=5,  # 评审人数
                    num_reviews_ensemble=1,  # 评审人数
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review.txt"), "w") as f:
                    f.write(json.dumps(review, indent=4))
            except Exception as e:
                debug_logger.error(f"Failed to perform review: {e}")
                return False

        ## 4. IMPROVE WRITEUP
        if writeup == "latex" and improvement:
            print_time()
            debug_logger.info(f"*Starting Improvement*")
            try:
                perform_improvement(review, coder)
                generate_latex(
                    coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
                )
                paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
                # 为什么又review了一次？
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                    f.write(json.dumps(review))
            except Exception as e:
                debug_logger.error(f"Failed to perform improvement: {e}")
                return False
        return True
    except Exception as e:
        debug_logger.error(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        debug_logger.info("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


if __name__ == "__main__":
    args = parse_arguments()

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(
            f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        args.parallel = len(available_gpus)

    # Create client
    if args.model == "claude-3-5-sonnet-20240620":
        import anthropic

        client_model = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic()
    elif args.model.startswith("bedrock") and "claude" in args.model:
        import anthropic

        # Expects: bedrock/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
    elif args.model.startswith("vertex_ai") and "claude" in args.model:
        import anthropic

        # Expects: vertex_ai/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        client = anthropic.AnthropicVertex()
    elif args.model == "gpt-4o-2024-05-13":
        import openai

        api_key = os.getenv('OPENAI_API_KEY_MTU')
        if api_key is None:
            raise ValueError("OPENAI_API_KEY_MTU environment variable not set.")

        base_url = os.getenv('OPENAI_API_BASE_URL_MTU')
        if base_url is None:
            raise ValueError("OPENAI_API_BASE_URL_MTU environment variable not set.")

        client_model = "gpt-4o-2024-05-13"
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    elif args.model == "deepseek-coder-v2-0724":
        import openai

        client_model = "deepseek-coder-v2-0724"
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    elif args.model == "llama3.1-405b":
        import openai

        client_model = "meta-llama/llama-3.1-405b-instruct"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Model {args.model} not supported.")

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    log_id = uuid.uuid4().hex
    # 将log_id设置为环境变量，以便在其他地方使用
    # os.environ["LOG_ID"] = log_id

    print(
        f"Using GPUs: {available_gpus}, parallel processes: {args.parallel}, base_dir: {base_dir}, results_dir: {results_dir}, log_id: {log_id}")

    debug_logger.info(f'log_id: {log_id}, will start generating ideas')
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
        log_id=log_id,
    )
    debug_logger.info(f'log_id: {log_id}, finished generating ideas, ideas: {ideas}')

    debug_logger.info(f'log_id: {log_id}, will start checking idea novelty')
    ideas = check_idea_novelty(
        ideas,
        base_dir=base_dir,
        client=client,
        model=client_model,
        max_num_iterations=3,  # 为了节省时间，这里暂时设置为3
        log_id=log_id,
    )

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    debug_logger.info(f'log_id: {log_id}, now finished checking idea novelty, ideas: {ideas}')

    # novel_ideas = [idea for idea in ideas if idea["novel"]]
    novel_ideas = ideas  # 暂时不过滤novelty
    # novel_ideas = list(reversed(novel_ideas))

    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes")
        queue = multiprocessing.Queue()
        for idea in novel_ideas:
            queue.put(idea)

        processes = []
        for i in range(args.parallel):
            gpu_id = available_gpus[i % len(available_gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(
                    queue,
                    base_dir,
                    results_dir,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                    gpu_id,
                ),
            )
            p.start()
            time.sleep(150)
            processes.append(p)

        # Signal workers to exit
        for _ in range(args.parallel):
            queue.put(None)

        for p in processes:
            p.join()

        print("All parallel processes completed.")
    else:
        for idea in novel_ideas:
            print(f"Processing idea: {idea['Name']}")
            try:
                success = do_idea(
                    base_dir,
                    results_dir,
                    idea,
                    args.model,
                    client,
                    client_model,
                    args.writeup,
                    args.improvement,
                )
                print(f"Completed idea: {idea['Name']}, Success: {success}")
            except Exception as e:
                print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")

    print("All ideas evaluated.")
