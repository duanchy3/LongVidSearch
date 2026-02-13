# LongVidSearch: An Agentic Benchmark for Multi-hop Evidence Retrieval Planning in Long Videos

> **LongVidSearch** evaluates **retrieval-necessary** and **evidence-grounded** multi-hop question answering over **untrimmed long videos** under a **standardized tool interface**, enabling controlled comparison of *agentic retrieval planning* across LLM backbones.

---

## Overview Figures

### Main Figure (Benchmark Framework)
<!-- TODO: replace with your main framework figure path -->
<p align="center">
  <img src="figs\mm-retrieval.pdf" width="92%" alt="LongVidSearch benchmark framework"/>
</p>
<p align="center">
  <em>Figure 1: Overview of LongVidSearch. Agents iteratively retrieve clips, read captions via standardized tools, and are evaluated by a three-judge majority vote protocol.</em>
</p>

### Data Figures (Dataset Statistics)
<!-- TODO: replace with your dataset statistics figure(s) -->
<p align="center">
  <img src="figs/stats_distribution.png" width="92%" alt="LongVidSearch dataset statistics"/>
</p>
<p align="center">
  <em>Figure 2: Dataset statistics of LongVidSearch (hop-level and category distributions).</em>
</p>

<!-- Optional: put multiple figures side-by-side -->
<!--
<p align="center">
  <img src="figs/hop_distribution.png" width="45%" alt="Hop distribution"/>
  <img src="figs/category_distribution.png" width="45%" alt="Category distribution"/>
</p>
<p align="center">
  <em>Figure 2: Hop distribution (left) and category distribution (right).</em>
</p>
-->

---

## What is LongVidSearch?

Long video question answering increasingly relies on **agentic tool use** to retrieve evidence from long videos. However, existing benchmarks rarely **standardize evidence access**, making it difficult to attribute failures to **retrieval planning** vs. **answer generation**.

**LongVidSearch** addresses this gap by:
- enforcing **retrieval necessity** (Hop-2/3/4, where each hop corresponds to a *necessary* evidence clip),
- requiring **evidence-grounded multi-hop reasoning** over long videos,
- providing a **unified tool interface** that fixes evidence access and the retrieval backend,
- reporting both **accuracy** and **tool-call cost** to study the **accuracy–cost trade-off**.

---

## Key Features

- **Retrieval-necessary multi-hop QA**: Hop-\(k\) questions require **\(k\) necessary evidence clips** (removing any one makes the question underdetermined).
- **Standardized tool interface**: identical evidence access for all agents to isolate **query formulation** and **multi-step evidence acquisition** capability.
- **Stable evaluation**: majority vote of **three strong LLM judges** (e.g., GPT-5 / Gemini 3 Pro / GPT-4o) with expert audit for consistency checking.
- **Efficiency-aware**: reports **tool-call cost** as a direct measure of evidence-access overhead.

---

## Dataset

- **3,159 QA pairs** from **447 long-form videos**
- Average video duration: **~26 minutes**
- Four capability categories:
  - **State Mutation (Entity + Transition)**: detect **critical transition points** and contrast pre/post states.
  - **Visual Tracking (Entity + Aggregation)**: aggregate appearances for **long-term ReID** across gaps/occlusions/view changes.
  - **Causal Inference (Narrative + Transition)**: establish a **semantic bridge** between cause and effect events.
  - **Global Summary (Narrative + Aggregation)**: synthesize a **holistic conclusion** from dispersed narrative evidence.

---

## Standardized Tools

All agents interact with LongVidSearch through the same tools:

- `Search_Clips_In_Video(video_id, query, top_k)`  
  Retrieves top-\(K\) relevant clips for a textual query within a given video.

- `Get_Clip_Detail(clip_id)`  
  Returns a high-quality caption for the queried clip (used as evidence).

- `FINAL_ANSWER(answer_text, evidence_clip_ids)`  
  Submits the answer and the list of viewed evidence clip IDs; evaluation computes accuracy and aggregates tool-call cost from logs.

This fixed interface ensures performance differences primarily reflect **agentic retrieval planning**, not retriever strength or privileged evidence access.

---

## Baseline Agent

We provide a VideoAgent-style baseline that follows an iterative **plan → retrieve → read → reason** loop:
1. generate a textual query based on current hypothesis and partial evidence,
2. retrieve candidate clips via `Search_Clips_In_Video`,
3. read captions via `Get_Clip_Detail`,
4. decide whether additional retrieval is needed,
5. output `FINAL_ANSWER` with selected evidence clip IDs.

> The baseline workflow diagram and prompt template can be found in the Appendix of the paper.

---

## Evaluation

### Metrics
- **Answer Accuracy**  
  Exact match where applicable; otherwise **LLM-as-a-judge** with a strict rubric and **three-judge majority vote**.

- **Tool-call Cost**  
  Number of standardized tool invocations per question, measuring evidence-access overhead.

### Oracle (Golden Clips)
We also include an oracle-style setting where the agent is given **golden evidence clips**. Near-perfect oracle accuracy indicates that the main bottleneck in the standard setting is **retrieval and retrieval planning**, rather than reasoning with correct evidence.

---

## Repository Structure (Suggested)

```text
.
├── data/
│   ├── longvidsearch.jsonl          # main QA file (example)
│   ├── splits/                      # train/val/test splits (if provided)
│   └── metadata/                    # video metadata, category/hop annotations
├── tools/
│   ├── search_clips_in_video.py     # retrieval tool wrapper
│   ├── get_clip_detail.py           # caption tool wrapper
│   └── interface.py                 # unified tool interface
├── baselines/
│   ├── agent.py                     # baseline VideoAgent-style framework
│   ├── prompts/                     # prompt templates
│   └── run_eval.py                  # evaluation runner
├── eval/
│   ├── judges/                      # judge prompts/rubric
│   ├── score.py                     # majority vote + metrics
│   └── cost.py                      # tool-call cost aggregation
├── figs/
│   ├── benchmark_framework-4.png    # main figure for README
│   └── stats_distribution.png       # data/stats figure for README
├── LICENSE
└── README.md
