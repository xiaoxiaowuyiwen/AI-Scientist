import json
import os.path as osp
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired

from ai_scientist.logger import debug_logger

MAX_ITERS = 4
# MAX_ITERS = 2  # 为了测试，减少实验次数

# MAX_RUNS = 5
MAX_RUNS = 2  # 为了测试，减少实验次数

MAX_STDERR_OUTPUT = 1500

coder_prompt = """Your goal is to implement the following idea: {title}.
The proposed experiment is as follows: {idea}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

After you complete each change, we will run the command `python experiment.py --out_dir=run_i' where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
You can then implement the next thing on your list."""


# RUN EXPERIMENT
# 注意这里的experiment.py是在folder_name文件夹下的experiment.py，是动态生成的
def run_experiment(folder_name, run_num, timeout=7200) -> (int, str):
    """
    执行实验

    :param folder_name: 目录名
    :param run_num: 实验编号
    :param timeout: 实验执行超时时间
    :return: 状态码和下一个prompt
    """
    cwd = osp.abspath(folder_name)
    # COPY CODE SO WE CAN SEE IT.
    shutil.copy(
        osp.join(folder_name, "experiment.py"),
        osp.join(folder_name, f"run_{run_num}.py"),
    )

    # LAUNCH COMMAND
    command = [
        "python",
        "experiment.py",
        f"--out_dir=run_{run_num}",
    ]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            debug_logger.error(f'run_experiment, result.stderr: {result.stderr}')
            # print(result.stderr, file=sys.stderr)

        if result.returncode != 0:  # 如果返回值不为0，说明实验执行失败
            debug_logger.info(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))  # 删除实验结果，避免影响下次实验？
            debug_logger.error(f"Run failed with the following error {result.stderr}")
            stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:  # 如果stderr输出太长，只保留后面的部分
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"  # 将错误信息返回给coder
        else:  # 如果返回值为0，说明实验执行成功
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)
            results = {k: v["means"] for k, v in results.items()}

            # 这个prompt指定了python experiment.py --out_dir=run_{run_num}，也就是在跑的过程中看到的终端输出
            next_prompt = f"""Run {run_num} completed. Here are the results:
{results}

Decide if you need to re-plan your experiments given the result (you often will not need to).

Someone else will be using `notes.txt` to perform a writeup on this in the future.
Please include *all* relevant information for the writeup on Run {run_num}, including an experiment description and the run number. Be as verbose as necessary.

Then, implement the next thing on your list.
We will then run the command `python experiment.py --out_dir=run_{run_num + 1}'.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
If you are finished with experiments, respond with 'ALL_COMPLETED'."""
        return result.returncode, next_prompt
    except TimeoutExpired:  # 如果实验超时
        debug_logger.error(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt


# RUN PLOTTING
def run_plotting(folder_name, timeout=600):
    cwd = osp.abspath(folder_name)
    # LAUNCH COMMAND
    command = [
        "python",
        "plot.py",
    ]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout
        )

        if result.stderr:
            print(result.stderr, file=sys.stderr)
            debug_logger.error(f'run_plotting, result.stderr: {result.stderr}')

        if result.returncode != 0:
            debug_logger.info(f"Plotting failed with return code {result.returncode}")
            next_prompt = f"Plotting failed with the following error {result.stderr}"
        else:
            next_prompt = ""
        return result.returncode, next_prompt
    except TimeoutExpired:
        debug_logger.error(f"Plotting timed out after {timeout} seconds")
        next_prompt = f"Plotting timed out after {timeout} seconds"
        return 1, next_prompt


# PERFORM EXPERIMENTS
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    ## RUN EXPERIMENT
    current_iter = 0
    run = 1
    next_prompt = coder_prompt.format(
        title=idea["Title"],
        idea=idea["Experiment"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    while run < MAX_RUNS + 1:
        # 这里实际上蕴含了2层循环，外层循环是run < MAX_RUNS + 1，内层循环是current_iter < MAX_ITERS
        # 如果当前实验失败，run是不会增加的，会继续尝试当前实验，直到成功或者

        if current_iter >= MAX_ITERS:  # 退出条件1：某次实验超过最大迭代次数
            debug_logger.info("Max iterations reached")
            break

        debug_logger.info(f'in perform_experiments, run: {run}, next_prompt: {next_prompt}')
        coder_out = coder.run(next_prompt)
        debug_logger.info(f'in perform_experiments, run: {run}, coder_out: {coder_out}')
        # print(coder_out)
        if "ALL_COMPLETED" in coder_out:  # 退出条件2：所有实验完成
            break
        # experiment.py是在什么时候生成的？
        return_code, next_prompt = run_experiment(folder_name, run)
        debug_logger.info(
            f'in perform_experiments, after run_experiment, run: {run}, return_code: {return_code}, next_prompt: {next_prompt}')
        if return_code == 0:  # 当前实验完成，准备进行下一个实验
            run += 1
            current_iter = 0
        current_iter += 1

    if current_iter >= MAX_ITERS:  # 不是所有实验都完成，直接返回False
        debug_logger.info("Not all experiments completed.")
        return False

    current_iter = 0

    # 画图
    debug_logger.info(f'now will modify plot.py and run_plotting')
    next_prompt = """
Great job! Please modify `plot.py` to generate the most relevant plots for the final writeup. 

In particular, be sure to fill in the "labels" dictionary with the correct names for each run that you want to plot.

Only the runs in the `labels` dictionary will be plotted, so make sure to include all relevant runs.

We will be running the command `python plot.py` to generate the plots.
"""
    while True:
        coder_out = coder.run(next_prompt)  # 修改plot.py

        return_code, next_prompt = run_plotting(folder_name)  # 画图
        current_iter += 1
        if return_code == 0:
            break
        if current_iter >= MAX_ITERS:
            debug_logger.error("Max iterations reached for plotting")
            break  # 原始代码这里也是break，为什么不直接返回False？

    # 修改notes.txt
    debug_logger.info(f'now will modify notes.txt')
    next_prompt = """
Please modify `notes.txt` with a description of what each plot shows along with the filename of the figure. Please do so in-depth.

Somebody else will be using `notes.txt` to write a report on this in the future.
"""
    coder.run(next_prompt)

    return True
