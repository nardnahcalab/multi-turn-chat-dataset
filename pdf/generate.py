#!/usr/bin/env python3
"""
Synthetic multi-turn PDF conversation generator for inference benchmarking.

Downloads arXiv paper metadata, then generates realistic multi-turn conversations
where users discuss PDF papers with an AI assistant. The PDF is referenced via
its arXiv URL in the first message, simulating real-world document analysis
workflows that stress prefix caching in LLM inference engines.

Usage:
    python generate.py                          # uses default config.yaml
    python generate.py --config my.yaml         # custom config
    python generate.py --num 1000 --seed 123    # override count and seed
    python generate.py --format aiperf          # only aiperf JSONL output
    python generate.py --skip-fetch             # reuse cached arxiv_papers.json
"""

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Add project root to path for shared module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_profile import (
    build_descriptive_name,
    build_manifest,
    print_profile_summary,
    save_manifest,
)

# ---------------------------------------------------------------------------
# arXiv paper fetcher
# ---------------------------------------------------------------------------

def fetch_arxiv_papers(config: dict, cache_path: Path) -> list[dict]:
    """Fetch paper metadata from arXiv API and cache locally."""
    if cache_path.exists():
        print(f"Loading cached papers from {cache_path}")
        with open(cache_path) as f:
            papers = json.load(f)
        if len(papers) >= config["papers"]["count"]:
            return papers[:config["papers"]["count"]]
        print(f"Cache has {len(papers)} papers, need {config['papers']['count']}. Fetching more...")

    import arxiv

    categories = config["papers"]["categories"]
    max_per_cat = config["papers"]["max_results_per_category"]
    target = config["papers"]["count"]

    all_papers = []
    seen_ids = set()

    for cat in categories:
        print(f"Fetching papers from {cat}...")
        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=max_per_cat,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
        for result in client.results(search):
            if result.entry_id in seen_ids:
                continue
            seen_ids.add(result.entry_id)

            paper = {
                "arxiv_id": result.entry_id.split("/abs/")[-1],
                "title": result.title.strip().replace("\n", " "),
                "authors": [a.name for a in result.authors[:10]],
                "abstract": result.summary.strip().replace("\n", " "),
                "categories": [c for c in result.categories],
                "primary_category": result.primary_category,
                "published": result.published.isoformat() if result.published else None,
                "pdf_url": result.pdf_url,
                "entry_url": result.entry_id,
            }
            all_papers.append(paper)

            if len(all_papers) >= target:
                break
        if len(all_papers) >= target:
            break
        time.sleep(1)  # be polite to arXiv API

    all_papers = all_papers[:target]
    print(f"Fetched {len(all_papers)} papers total")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"Cached to {cache_path}")

    return all_papers


# ---------------------------------------------------------------------------
# Conversation templates for PDF analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert AI research assistant. You help users understand, analyze, "
    "and discuss academic papers. You provide thorough, accurate analysis of paper "
    "content including methodology, results, and implications. When referencing "
    "specific sections, be precise about what the paper states versus your interpretation."
)

CONVERSATION_TEMPLATES = {
    "paper_summary": {
        "openers": [
            "I just found this paper: \"{title}\" by {authors}. Can you give me a comprehensive summary? Here's the PDF: {pdf_url}",
            "Can you read through this paper and summarize the key contributions? {pdf_url}\n\nTitle: \"{title}\"",
            "I need to quickly understand what this paper is about. Please summarize the main ideas, methods, and results.\n\nPaper: \"{title}\" ({pdf_url})",
            "I'm reviewing papers for a literature survey. Can you help me understand this one?\n\n\"{title}\" by {first_author} et al.\n{pdf_url}",
            "What are the main takeaways from this paper? {pdf_url}\n\n\"{title}\"",
        ],
        "followups": [
            "Can you explain the main contribution in simpler terms? I'm not an expert in {field}.",
            "What problem are they trying to solve, and why is it important?",
            "How does their approach differ from previous work in this area?",
            "Can you break down the abstract for me? Some of the terminology is unfamiliar.",
            "What are the key claims the authors make?",
            "Is this paper building on any specific prior work? What's the lineage?",
            "Who would benefit most from reading this paper?",
            "Can you summarize each section briefly?",
            "What's the most novel aspect of this work?",
            "How does this fit into the broader landscape of {field} research?",
        ],
        "responses": [
            "Here's a comprehensive summary of \"{title}\":\n\n**Problem:** The paper addresses {problem_description}. This is an important challenge because {importance}.\n\n**Approach:** The authors propose {approach_description}. The key insight is {key_insight}.\n\n**Key Contributions:**\n1. {contribution_1}\n2. {contribution_2}\n3. {contribution_3}\n\n**Results:** {results_summary}\n\n**Significance:** This work advances the state of the art by {advancement}.",
            "This paper makes several important contributions to {field}:\n\n**Main Idea:** {approach_description}\n\n**Why It Matters:** {importance}\n\nThe authors demonstrate that {key_finding}. Compared to prior approaches, their method {comparison}.\n\nThe experimental evaluation covers {eval_description}, showing {results_summary}.",
            "Let me break this down:\n\n**TL;DR:** {tldr}\n\n**The Problem:** {problem_description}\n\n**Their Solution:** {approach_description}\n\n**How It Works:**\n{method_steps}\n\n**Results Highlights:**\n- {result_1}\n- {result_2}\n- {result_3}\n\n**Bottom Line:** {bottom_line}",
        ],
    },

    "methodology_deep_dive": {
        "openers": [
            "I'm reading \"{title}\" ({pdf_url}) and I want to understand the methodology in detail. Can you walk me through their approach?",
            "Can you explain the technical approach in this paper? I'm particularly interested in how they {method_aspect}.\n\nPaper: {pdf_url}",
            "I need to understand the architecture/algorithm described in \"{title}\" ({pdf_url}). Can you break it down step by step?",
            "What's the core technical contribution of this paper? {pdf_url}\n\n\"{title}\" by {authors}",
        ],
        "followups": [
            "Can you go deeper into the {component} component? How exactly does it work?",
            "What's the intuition behind using {technique} here?",
            "How does the training procedure work? What's the loss function?",
            "What are the hyperparameters and how sensitive is the method to them?",
            "Can you explain the mathematical formulation in Section {section}?",
            "How does this compare to {alternative_method} in terms of approach?",
            "What are the computational requirements? Is this practical at scale?",
            "Are there any assumptions that might limit the applicability?",
            "How would you implement this from scratch? What are the key steps?",
            "What data preprocessing or augmentation do they use?",
            "Is the model architecture novel or is it an adaptation of existing work?",
            "Can you explain the attention mechanism / loss function / optimization they describe?",
            "What would happen if you changed {component} to use {alternative}?",
        ],
        "responses": [
            "The methodology in \"{title}\" consists of several key components:\n\n**1. {component_1_name}**\n{component_1_detail}\n\n**2. {component_2_name}**\n{component_2_detail}\n\n**3. {component_3_name}**\n{component_3_detail}\n\n**Training:**\n{training_detail}\n\n**Key Design Choices:**\n- {design_choice_1}\n- {design_choice_2}\n\nThe intuition behind this approach is {intuition}.",
            "Let me walk through the technical details:\n\n**Architecture Overview:**\n{architecture_overview}\n\n**The core innovation is {innovation}**, which works by:\n1. {step_1}\n2. {step_2}\n3. {step_3}\n\n**Formally**, the authors define {formal_description}.\n\nThis is significant because {significance}. In practice, this means {practical_implication}.",
            "Great question about {component}. Here's how it works:\n\n{detailed_explanation}\n\nThe key insight is that {key_insight}. This is different from standard approaches because {difference}.\n\n**Complexity:** {complexity_analysis}\n\n**Trade-offs:** {tradeoffs}",
        ],
    },

    "results_analysis": {
        "openers": [
            "Can you analyze the experimental results in \"{title}\"? ({pdf_url}) I want to understand how well their approach actually performs.",
            "I'm looking at the results section of this paper ({pdf_url}). Can you help me interpret the numbers and figures?",
            "How strong are the experimental results in \"{title}\"? {pdf_url}",
            "Walk me through the evaluation in this paper. What benchmarks did they use and how did they perform? {pdf_url}",
        ],
        "followups": [
            "How do these results compare to the current state of the art?",
            "Are the improvements statistically significant?",
            "What benchmarks/datasets did they evaluate on?",
            "Can you interpret Table {table_num} for me?",
            "What about the ablation study? Which components matter most?",
            "Are there any cases where their method performs poorly?",
            "How fair is the comparison with baselines?",
            "What metrics are they using and are they the right ones?",
            "Do the qualitative results support the quantitative findings?",
            "How would this perform on out-of-distribution data?",
            "What's the gap between their method and the theoretical upper bound?",
            "Are there any red flags in how the experiments were conducted?",
        ],
        "responses": [
            "Here's my analysis of the experimental results:\n\n**Benchmarks Used:** {benchmarks}\n\n**Main Results:**\n{main_results_table}\n\n**Key Findings:**\n1. {finding_1}\n2. {finding_2}\n3. {finding_3}\n\n**Ablation Analysis:**\n{ablation_summary}\n\n**Strengths of the evaluation:** {eval_strengths}\n**Potential concerns:** {eval_concerns}",
            "The results show {overall_assessment}:\n\n**Quantitative Performance:**\n- On {benchmark_1}: {performance_1}\n- On {benchmark_2}: {performance_2}\n- On {benchmark_3}: {performance_3}\n\n**Compared to baselines**, the proposed method {comparison_summary}.\n\n**The ablation study reveals** that {ablation_key_finding}. This suggests {implication}.\n\n**My assessment:** {assessment}",
        ],
    },

    "critical_review": {
        "openers": [
            "Can you give me a critical review of \"{title}\"? ({pdf_url}) I want to understand both the strengths and weaknesses.",
            "I'm trying to decide if the claims in this paper are well-supported. Can you help me evaluate it critically? {pdf_url}",
            "What would a peer reviewer say about this paper? {pdf_url}\n\n\"{title}\" by {authors}",
            "Help me assess the quality and impact of this paper: \"{title}\" ({pdf_url})",
        ],
        "followups": [
            "What are the strongest aspects of this work?",
            "What are the most significant limitations?",
            "Are there any methodological issues you noticed?",
            "How reproducible do you think this work is?",
            "Is the related work section thorough?",
            "Do the conclusions follow from the evidence presented?",
            "What experiments are missing that would strengthen the paper?",
            "How novel is this really, compared to prior work?",
            "Would you accept this paper at a top venue? Why or why not?",
            "What ethical considerations should the authors have addressed?",
            "Is the paper well-written and clearly structured?",
        ],
        "responses": [
            "Here's my critical assessment of \"{title}\":\n\n**Strengths:**\n- {strength_1}\n- {strength_2}\n- {strength_3}\n\n**Weaknesses:**\n- {weakness_1}\n- {weakness_2}\n- {weakness_3}\n\n**Missing Elements:**\n{missing_elements}\n\n**Novelty Assessment:** {novelty_assessment}\n\n**Reproducibility:** {reproducibility_assessment}\n\n**Overall:** {overall_verdict}",
            "This is a {quality_level} paper with some notable {notable_aspect}:\n\n**What works well:**\n{strengths_detail}\n\n**What could be improved:**\n{weaknesses_detail}\n\n**Key question for the authors:** {key_question}\n\n**Impact potential:** {impact_assessment}",
        ],
    },

    "comparison": {
        "openers": [
            "How does the approach in \"{title}\" ({pdf_url}) compare to other methods in {field}?",
            "I've been reading several papers on {field}. Can you help me understand where \"{title}\" fits in? {pdf_url}",
            "What makes this paper different from {related_method}? {pdf_url}\n\n\"{title}\"",
            "Can you compare the approach in this paper with the current state of the art? {pdf_url}",
        ],
        "followups": [
            "What are the trade-offs between this approach and {alternative_method}?",
            "Which method would you recommend for {use_case}?",
            "How does the computational cost compare?",
            "Are there scenarios where the older approaches would still be preferred?",
            "What did this paper borrow from prior work and what's genuinely new?",
            "How does the evaluation setup differ from related papers?",
            "Can you trace the evolution of ideas that led to this paper?",
            "Which baseline is the most informative comparison?",
            "How would you combine the best ideas from these different approaches?",
        ],
        "responses": [
            "Here's how \"{title}\" compares to related work:\n\n**Comparison Matrix:**\n\n| Aspect | This Paper | {method_a} | {method_b} |\n|--------|-----------|------------|------------|\n| Approach | {this_approach} | {approach_a} | {approach_b} |\n| Performance | {this_perf} | {perf_a} | {perf_b} |\n| Efficiency | {this_eff} | {eff_a} | {eff_b} |\n| Limitations | {this_limit} | {limit_a} | {limit_b} |\n\n**Key Differences:**\n{key_differences}\n\n**The main advantage** of this paper's approach is {main_advantage}. However, {caveat}.",
            "Positioning \"{title}\" in the broader landscape:\n\n**Evolution of approaches:**\n1. {era_1}: {approach_era_1}\n2. {era_2}: {approach_era_2}\n3. **This paper**: {this_contribution}\n\n**What's genuinely novel:** {novelty}\n\n**What's borrowed:** {borrowed_elements}\n\n**Trade-offs vs. alternatives:**\n- vs. {method_a}: {tradeoff_a}\n- vs. {method_b}: {tradeoff_b}\n\n**Recommendation:** {recommendation}",
        ],
    },

    "implementation": {
        "openers": [
            "I want to implement the approach from \"{title}\" ({pdf_url}). What do I need to know?",
            "Is there enough detail in this paper to reproduce the results? {pdf_url}\n\n\"{title}\"",
            "What are the key implementation details for the method described in \"{title}\"? {pdf_url}",
            "I'm a practitioner looking to use this in production. What should I know about implementing \"{title}\"? {pdf_url}",
        ],
        "followups": [
            "What framework/libraries would you recommend for implementation?",
            "What hardware would I need to train this model?",
            "Are there any open-source implementations available?",
            "What are the most common pitfalls when implementing this?",
            "How long would training take on {hardware}?",
            "What dataset would I need and how should I preprocess it?",
            "Are there any tricks for making training stable?",
            "How would I adapt this for my specific use case in {domain}?",
            "What monitoring/debugging strategies would you suggest?",
            "How would I evaluate whether my implementation is correct?",
        ],
        "responses": [
            "Here's an implementation guide for \"{title}\":\n\n**Prerequisites:**\n- {prereq_1}\n- {prereq_2}\n- {prereq_3}\n\n**Key Implementation Steps:**\n1. {impl_step_1}\n2. {impl_step_2}\n3. {impl_step_3}\n4. {impl_step_4}\n\n**Critical Details:**\n{critical_details}\n\n**Common Pitfalls:**\n- {pitfall_1}\n- {pitfall_2}\n\n**Estimated Resources:**\n- Training: {training_resources}\n- Inference: {inference_resources}",
            "For implementing this approach, here's what matters most:\n\n**Architecture:**\n{architecture_guide}\n\n**Training Recipe:**\n{training_recipe}\n\n**Data Pipeline:**\n{data_pipeline}\n\n**Hyperparameters to tune:**\n{hyperparam_guide}\n\n**Validation strategy:** {validation_strategy}\n\n**Reproducibility notes:** {repro_notes}",
        ],
    },

    "brainstorm_extensions": {
        "openers": [
            "After reading \"{title}\" ({pdf_url}), I'm thinking about possible extensions. What improvements or follow-up research do you see?",
            "What are the most promising research directions building on \"{title}\"? {pdf_url}",
            "If you were to write a follow-up paper to \"{title}\", what would you investigate? {pdf_url}",
            "How could the approach in this paper be improved or extended? {pdf_url}\n\n\"{title}\"",
        ],
        "followups": [
            "Can you elaborate on the {extension} idea? How would that work?",
            "What would the experimental setup look like for testing that extension?",
            "Are there other domains where this approach could be applied?",
            "What's the lowest-hanging fruit for improvement?",
            "How could this be combined with {other_technique}?",
            "What are the biggest open questions left by this paper?",
            "Is there a way to make this more efficient without sacrificing quality?",
            "How might this approach evolve in the next 2-3 years?",
            "What would a production-ready version of this look like?",
            "Could this be applied to {application_domain}?",
        ],
        "responses": [
            "Based on \"{title}\", here are the most promising extensions:\n\n**1. {extension_1_name}**\n{extension_1_detail}\nFeasibility: {feasibility_1}\n\n**2. {extension_2_name}**\n{extension_2_detail}\nFeasibility: {feasibility_2}\n\n**3. {extension_3_name}**\n{extension_3_detail}\nFeasibility: {feasibility_3}\n\n**Open Questions:**\n- {open_question_1}\n- {open_question_2}\n\n**What I'd prioritize:** {priority_recommendation}",
            "Great question. Here's how I'd think about extending this work:\n\n**Short-term improvements:**\n{short_term}\n\n**Medium-term research directions:**\n{medium_term}\n\n**Long-term vision:**\n{long_term}\n\n**Cross-pollination opportunities:**\n{cross_pollination}\n\n**The key bottleneck** to address is {bottleneck}. If solved, it would unlock {unlock}.",
        ],
    },
}

# Fill values for template placeholders (AI/ML domain specific)
FILL_VALUES = {
    "field": [
        "natural language processing", "computer vision", "reinforcement learning",
        "generative models", "graph neural networks", "multimodal learning",
        "large language models", "self-supervised learning", "federated learning",
        "neural architecture search", "model compression", "adversarial robustness",
    ],
    "technique": [
        "attention mechanisms", "contrastive learning", "knowledge distillation",
        "mixture of experts", "diffusion models", "prompt tuning",
        "low-rank adaptation", "chain-of-thought reasoning", "retrieval augmentation",
    ],
    "component": [
        "encoder", "decoder", "attention layer", "embedding module", "loss function",
        "data augmentation pipeline", "normalization strategy", "positional encoding",
        "gating mechanism", "routing layer", "tokenizer",
    ],
    "alternative_method": [
        "standard fine-tuning", "the baseline transformer", "a CNN-based approach",
        "simple retrieval", "prompt engineering", "LoRA", "full fine-tuning",
        "RLHF", "DPO", "constitutional AI",
    ],
    "related_method": [
        "GPT-style autoregressive models", "BERT-family models", "Vision Transformers",
        "diffusion-based methods", "GAN-based approaches", "neuro-symbolic methods",
    ],
    "use_case": [
        "a production chatbot", "real-time inference", "edge deployment",
        "a large-scale data pipeline", "a low-resource language",
        "medical image analysis", "code generation",
    ],
    "hardware": [
        "a single A100 GPU", "8x H100 GPUs", "a consumer RTX 4090",
        "a TPU v4 pod", "4x A6000 GPUs",
    ],
    "domain": [
        "healthcare", "finance", "autonomous driving", "robotics",
        "scientific discovery", "content moderation", "education",
    ],
    "section": ["3", "4", "3.2", "4.1", "2.3", "5"],
    "table_num": ["1", "2", "3", "4"],
    "extension": [
        "scaling to more modalities", "reducing computational cost",
        "improving few-shot performance", "better evaluation metrics",
        "cross-lingual transfer", "online/continual learning adaptation",
    ],
    "other_technique": [
        "retrieval-augmented generation", "chain-of-thought prompting",
        "model merging", "speculative decoding", "tool use",
    ],
    "application_domain": [
        "drug discovery", "climate modeling", "code review automation",
        "educational tutoring", "legal document analysis",
    ],
    "method_aspect": [
        "handle long sequences", "reduce hallucinations", "scale to larger models",
        "incorporate external knowledge", "handle multimodal inputs",
    ],
    "problem_description": [
        "the challenge of scaling transformer models to longer context windows while maintaining efficiency",
        "the difficulty of aligning model outputs with human preferences without extensive annotation",
        "the limitation of existing methods that struggle with distribution shift at inference time",
        "the need for more sample-efficient learning in low-resource scenarios",
        "the gap between pre-training objectives and downstream task performance",
    ],
    "importance": [
        "it directly impacts the real-world usability of large language models",
        "current approaches are prohibitively expensive for most practitioners",
        "this bottleneck limits progress across multiple downstream applications",
        "robust solutions here would enable a new class of applications",
    ],
    "approach_description": [
        "a novel architecture that combines efficient attention with hierarchical representations",
        "a training framework that leverages self-supervised objectives with targeted fine-tuning",
        "a method that reformulates the problem as an optimization over a learned latent space",
        "an approach that uses synthetic data generation to augment limited training examples",
        "a pipeline that decomposes the task into modular stages with specialized components",
    ],
    "key_insight": [
        "that structured sparsity patterns can approximate full attention with much lower cost",
        "that the right inductive bias can dramatically reduce the data needed for generalization",
        "that iterative refinement outperforms single-shot prediction for complex tasks",
        "that cross-modal alignment can be achieved through shared representation spaces",
    ],
    "contribution_1": [
        "A novel architecture for efficient long-range sequence modeling",
        "A training methodology that reduces compute requirements by 3-5x",
        "New theoretical analysis establishing bounds on the approach's effectiveness",
        "A comprehensive benchmark for evaluating methods in this space",
    ],
    "contribution_2": [
        "State-of-the-art results on multiple standard benchmarks",
        "An ablation study demonstrating the importance of each component",
        "A practical open-source implementation with reproducibility guarantees",
        "Analysis of failure modes and edge cases",
    ],
    "contribution_3": [
        "Insights into the scaling behavior of the proposed method",
        "A thorough comparison with 10+ baseline methods",
        "Transfer learning experiments showing generalization across domains",
        "Efficiency analysis showing favorable speed-accuracy trade-offs",
    ],
    "results_summary": [
        "The method achieves a 15-20% improvement over the previous state of the art on standard benchmarks, while using 40% less compute",
        "Experiments across 5 benchmarks show consistent improvements, with the largest gains on the most challenging tasks",
        "The approach matches or exceeds existing methods while being significantly more efficient at inference time",
        "Results demonstrate strong performance in both in-distribution and out-of-distribution settings",
    ],
    "advancement": [
        "demonstrating that efficiency and effectiveness need not be at odds",
        "providing a principled framework for tackling this class of problems",
        "opening up new possibilities for practical deployment of these models",
        "establishing new baselines that future work can build upon",
    ],
    "tldr": [
        "They propose a more efficient way to do X that works as well as or better than existing approaches",
        "The paper introduces a new training paradigm that significantly improves sample efficiency",
        "A novel architecture achieves state-of-the-art results while being much faster",
        "They show that a simple modification to existing methods leads to substantial improvements",
    ],
    "method_steps": [
        "1. First, they encode the input using a modified transformer encoder\n2. Then, they apply their novel attention mechanism to capture long-range dependencies\n3. A task-specific head produces the final output\n4. The entire model is trained end-to-end with a composite loss",
        "1. Pre-process the data with their proposed augmentation strategy\n2. Train the base model with a self-supervised objective\n3. Fine-tune with a small amount of labeled data\n4. Apply their inference-time optimization for better performance",
    ],
    "result_1": ["15.3% improvement on the primary benchmark over the previous SOTA"],
    "result_2": ["2.5x faster inference speed with comparable accuracy"],
    "result_3": ["Strong performance maintained even with 10x less training data"],
    "bottom_line": [
        "A solid contribution that moves the field forward with practical improvements",
        "Promising approach with strong empirical results, though some limitations remain",
        "Well-executed work that provides both theoretical insights and practical tools",
    ],
    "quality_level": ["strong", "solid", "promising", "well-crafted"],
    "notable_aspect": ["strengths", "contributions", "insights"],
    "strength_1": ["Clear problem formulation and motivation"],
    "strength_2": ["Comprehensive experimental evaluation with strong baselines"],
    "strength_3": ["Good ablation study that isolates the contribution of each component"],
    "weakness_1": ["Limited evaluation on diverse domains beyond the primary benchmarks"],
    "weakness_2": ["Some implementation details are missing, which could affect reproducibility"],
    "weakness_3": ["The computational cost analysis could be more thorough"],
    "overall_verdict": [
        "A valuable contribution that would benefit from a few more experiments to strengthen the claims",
        "Solid work with clear merits, though the novelty is somewhat incremental",
        "Strong paper with both theoretical and practical contributions",
    ],
    "benchmarks": [
        "GLUE, SuperGLUE, and domain-specific benchmarks",
        "ImageNet, COCO, and custom task-specific datasets",
        "MMLU, HellaSwag, ARC, and TruthfulQA",
        "HumanEval, MBPP, and custom code generation benchmarks",
    ],
    "component_1_name": ["Feature Extraction Module"],
    "component_1_detail": ["Uses a hierarchical encoder to capture both local and global patterns in the input."],
    "component_2_name": ["Adaptive Processing Layer"],
    "component_2_detail": ["Dynamically adjusts computation based on input complexity, allocating more resources to harder examples."],
    "component_3_name": ["Output Generation Head"],
    "component_3_detail": ["Combines multiple representation levels to produce the final prediction with calibrated confidence."],
    "training_detail": ["The model is trained in two phases: pre-training on a large unlabeled corpus, followed by task-specific fine-tuning with a carefully designed curriculum."],
    "intuition": ["by decomposing the problem into manageable sub-tasks, each component can specialize without interference"],
    "innovation": ["a novel way to combine local and global information that avoids the quadratic cost of standard attention"],
    "architecture_overview": ["The model follows an encoder-decoder architecture with several modifications to improve efficiency and effectiveness."],
    "extension_1_name": ["Multi-modal Extension"],
    "extension_1_detail": ["Extend the approach to handle images, audio, and video alongside text by learning shared representations."],
    "extension_2_name": ["Efficiency Improvements"],
    "extension_2_detail": ["Apply quantization, pruning, and distillation to make the model practical for edge deployment."],
    "extension_3_name": ["Broader Evaluation"],
    "extension_3_detail": ["Test on more diverse benchmarks, especially low-resource and cross-lingual settings."],
    "feasibility_1": ["High — the architecture naturally supports this with minor modifications"],
    "feasibility_2": ["Medium — requires careful engineering but no fundamental barriers"],
    "feasibility_3": ["Low-effort — primarily a data/compute question"],
}


# ---------------------------------------------------------------------------
# Conversation generator
# ---------------------------------------------------------------------------

class PDFConversationGenerator:
    def __init__(self, config: dict, papers: list[dict], seed: int = 42):
        self.config = config
        self.papers = papers
        self.rng = random.Random(seed)
        self.conv_types = config["conversation_types"]
        self.conv_weights = [t["weight"] for t in self.conv_types]

    def _fill_template(self, template: str, paper: dict) -> str:
        """Fill placeholders with paper metadata and generic values."""
        result = template

        # Paper-specific fills
        author_str = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            author_str += " et al."
        paper_fills = {
            "{title}": paper["title"],
            "{authors}": author_str,
            "{first_author}": paper["authors"][0] if paper["authors"] else "the authors",
            "{pdf_url}": paper["pdf_url"],
            "{abstract}": paper["abstract"][:500],
            "{arxiv_id}": paper["arxiv_id"],
            "{primary_category}": paper["primary_category"],
        }
        for key, value in paper_fills.items():
            result = result.replace(key, value)

        # Generic fills
        import re
        max_iter = 10
        for _ in range(max_iter):
            if "{" not in result:
                break
            for key, values in FILL_VALUES.items():
                placeholder = "{" + key + "}"
                while placeholder in result:
                    result = result.replace(placeholder, self.rng.choice(values), 1)
            # Clean up any remaining unfilled placeholders
            result = re.sub(r'\{[a-zA-Z_0-9]+\}', '', result)

        return result.strip()

    def _response_length_bucket(self, turn_index: int) -> str:
        dist_config = self.config["response_length"]["length_distribution_by_turn"]
        if turn_index < 3:
            dist = dist_config["early"]
        elif turn_index < 15:
            dist = dist_config["middle"]
        else:
            dist = dist_config["late"]
        buckets = list(dist.keys())
        weights = list(dist.values())
        return self.rng.choices(buckets, weights=weights, k=1)[0]

    def _generate_response(self, conv_type: str, paper: dict, turn_index: int) -> str:
        """Generate an assistant response of appropriate length."""
        templates = CONVERSATION_TEMPLATES[conv_type]["responses"]
        base = self._fill_template(self.rng.choice(templates), paper)

        bucket = self._response_length_bucket(turn_index)
        length_config = self.config["response_length"][bucket]
        target_words = self.rng.randint(length_config["min_words"], length_config["max_words"])

        words = base.split()
        if len(words) < target_words:
            extensions = [
                f"\n\nIt's also worth noting that the paper's approach to {self.rng.choice(FILL_VALUES['technique'])} has broader implications for the field.",
                f"\n\nFrom a practical standpoint, this could significantly impact applications in {self.rng.choice(FILL_VALUES['domain'])}.",
                "\n\nI'd recommend reading Section 4 carefully, as it contains important implementation details that aren't immediately obvious from the abstract.",
                "\n\nOne thing to keep in mind is that these results were obtained under specific experimental conditions, and performance may vary in different settings.",
                "\n\nThe authors acknowledge several limitations, which is refreshing. The most significant is the reliance on large-scale compute for training.",
                f"\n\nCompared to {self.rng.choice(FILL_VALUES['alternative_method'])}, this approach offers a different trade-off between complexity and performance.",
                "\n\nFor anyone looking to build on this work, I'd suggest starting with the ablation study to understand which components are essential versus optional.",
                "\n\nThe theoretical analysis in the paper is particularly insightful, providing bounds that help explain when and why the method works well.",
            ]
            while len(words) < target_words:
                words.extend(self.rng.choice(extensions).split())
        elif len(words) > target_words * 1.3:
            words = words[:target_words]
            text = " ".join(words)
            last_period = text.rfind(".")
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]
            return text

        return " ".join(words)

    def _generate_user_message(self, conv_type: str, paper: dict, turn_index: int) -> str:
        templates = CONVERSATION_TEMPLATES[conv_type]
        if turn_index == 0:
            template = self.rng.choice(templates["openers"])
        else:
            template = self.rng.choice(templates["followups"])
        return self._fill_template(template, paper)

    def generate_conversation(self, num_turns: int) -> dict:
        """Generate a single multi-turn PDF conversation."""
        conv_type_cfg = self.rng.choices(self.conv_types, weights=self.conv_weights, k=1)[0]
        conv_type = conv_type_cfg["name"]
        paper = self.rng.choice(self.papers)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # First user message includes the PDF URL reference
        # Using OpenAI-compatible multimodal format for the first message
        first_user_text = self._generate_user_message(conv_type, paper, 0)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": first_user_text},
                {
                    "type": "file",
                    "file": {
                        "url": paper["pdf_url"],
                        "mime_type": "application/pdf",
                    },
                },
            ],
        })

        # First assistant response
        first_response = self._generate_response(conv_type, paper, 0)
        messages.append({"role": "assistant", "content": first_response})

        cumulative_char_lengths = []
        running_chars = sum(
            len(str(m.get("content", ""))) for m in messages
        )
        cumulative_char_lengths.append(running_chars)

        # Subsequent turns (text only — the PDF is already in context)
        for turn_idx in range(1, num_turns):
            user_msg = self._generate_user_message(conv_type, paper, turn_idx)
            messages.append({"role": "user", "content": user_msg})
            running_chars += len(user_msg)

            assistant_msg = self._generate_response(conv_type, paper, turn_idx)
            messages.append({"role": "assistant", "content": assistant_msg})
            running_chars += len(assistant_msg)

            cumulative_char_lengths.append(running_chars)

        conversation_id = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

        return {
            "conversation_id": conversation_id,
            "conversation_type": conv_type,
            "paper_arxiv_id": paper["arxiv_id"],
            "paper_title": paper["title"],
            "paper_pdf_url": paper["pdf_url"],
            "paper_categories": json.dumps(paper["categories"]),
            "num_turns": num_turns,
            "num_messages": len(messages),
            "system_prompt": SYSTEM_PROMPT,
            "messages": json.dumps(messages),
            "total_characters": running_chars,
            "estimated_tokens": running_chars // 4,
            "cumulative_char_lengths": json.dumps(cumulative_char_lengths),
        }

    def generate_dataset(self, num_conversations: int = None) -> list[dict]:
        if num_conversations is not None:
            conversations = []
            turn_min = self.config["turns"]["min"]
            turn_max = self.config["turns"]["max"]
            for _ in range(num_conversations):
                n_turns = self.rng.randint(turn_min, turn_max)
                conversations.append(self.generate_conversation(n_turns))
            return conversations

        conversations = []
        dist = self.config["turns"]["distribution"]
        for bucket_name, bucket_cfg in dist.items():
            count = bucket_cfg["count"]
            min_t = bucket_cfg["min_turns"]
            max_t = bucket_cfg["max_turns"]
            for _ in range(count):
                n_turns = self.rng.randint(min_t, max_t)
                conversations.append(self.generate_conversation(n_turns))

        self.rng.shuffle(conversations)
        return conversations


# ---------------------------------------------------------------------------
# aiperf export functions
# ---------------------------------------------------------------------------

def convert_to_aiperf_multi_turn(conversations: list[dict]) -> list[dict]:
    """Convert to aiperf multi_turn JSONL format.

    For PDF conversations, the first turn includes the PDF URL in the
    multimodal content format. Subsequent turns are text-only follow-ups.
    """
    entries = []
    for conv in conversations:
        messages = json.loads(conv["messages"])
        turns = []
        for msg in messages:
            if msg["role"] == "user":
                # Handle multimodal first message
                if isinstance(msg["content"], list):
                    # Extract text and file reference
                    text_parts = []
                    files = []
                    for part in msg["content"]:
                        if part.get("type") == "text":
                            text_parts.append(part["text"])
                        elif part.get("type") == "file":
                            files.append(part["file"]["url"])
                    text = " ".join(text_parts)
                    if files:
                        text += "\n\n[PDF: " + files[0] + "]"
                    turns.append({"text": text})
                else:
                    turns.append({"text": msg["content"]})
        if turns:
            entries.append({
                "session_id": conv["conversation_id"],
                "turns": turns,
            })
    return entries


def convert_to_aiperf_mooncake(conversations: list[dict]) -> list[dict]:
    """Convert to aiperf mooncake_trace JSONL with full message arrays."""
    entries = []
    for conv in conversations:
        messages = json.loads(conv["messages"])
        session_id = conv["conversation_id"]
        context = []
        for msg in messages:
            context.append(msg)
            if msg["role"] == "assistant":
                output_tokens = max(1, len(str(msg["content"])) // 4)
                entry = {
                    "session_id": session_id,
                    "messages": [m for m in context],
                    "output_length": output_tokens,
                }
                turn_index = len([m for m in context if m["role"] == "assistant"])
                if turn_index > 1:
                    entry["delay"] = 0
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-turn PDF chat dataset from arXiv papers"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--num", type=int, default=None, help="Override number of conversations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--format", choices=["all", "parquet", "aiperf", "mooncake"],
                        default="all", help="Output format(s)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip arXiv fetch, reuse cached papers")
    parser.add_argument("--name", default=None,
                        help="Custom suffix for descriptive output filenames")
    parser.add_argument("--descriptive-names", action="store_true", default=False,
                        help="Use descriptive filenames encoding count, seed, version, and date")
    parser.add_argument("--no-profile", action="store_true", default=False,
                        help="Skip generating the dataset manifest/profile JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config["dataset"]["seed"]
    cache_path = Path(__file__).parent / config["papers"]["cache_file"]

    # Fetch or load arXiv papers
    if args.skip_fetch and cache_path.exists():
        print(f"Loading cached papers from {cache_path}")
        with open(cache_path) as f:
            papers = json.load(f)
    else:
        papers = fetch_arxiv_papers(config, cache_path)

    print(f"Using {len(papers)} arXiv papers")

    generator = PDFConversationGenerator(config, papers, seed=seed)

    num_conversations = args.num
    print(f"Generating PDF conversations (seed={seed})...")
    conversations = generator.generate_dataset(num_conversations=num_conversations)
    print(f"Generated {len(conversations)} conversations")

    df = pd.DataFrame(conversations)

    print(f"\n--- PDF Dataset Summary ---")
    print(f"Total conversations: {len(df)}")
    print(f"Turn count range: {df['num_turns'].min()} - {df['num_turns'].max()}")
    print(f"Mean turns: {df['num_turns'].mean():.1f}")
    print(f"Unique papers used: {df['paper_arxiv_id'].nunique()}")
    print(f"Conversation type distribution:")
    for ctype, count in df["conversation_type"].value_counts().items():
        print(f"  {ctype}: {count} ({100*count/len(df):.1f}%)")
    print(f"Estimated total tokens: {df['estimated_tokens'].sum():,}")
    print(f"Mean tokens/conversation: {df['estimated_tokens'].mean():,.0f}")
    print(f"Max tokens (single conversation): {df['estimated_tokens'].max():,}")

    output_dir = (Path(args.output).parent if args.output
                  else Path(__file__).parent / config["dataset"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_count = len(df)
    descriptive_name = build_descriptive_name(
        config, actual_count, seed, "pdf", custom_suffix=args.name
    )

    if args.descriptive_names:
        file_base = descriptive_name
    else:
        file_base = config["dataset"]["output_filename"].replace(".parquet", "")

    fmt = args.format
    output_files = {}

    if fmt in ("all", "parquet"):
        parquet_path = (Path(args.output) if args.output and fmt == "parquet"
                        else output_dir / f"{file_base}.parquet")
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        mb = parquet_path.stat().st_size / (1024 * 1024)
        output_files["parquet"] = str(parquet_path)
        print(f"\nParquet written to: {parquet_path} ({mb:.2f} MB)")

    if fmt in ("all", "aiperf"):
        entries = convert_to_aiperf_multi_turn(conversations)
        jsonl_path = output_dir / f"{file_base}.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        mb = jsonl_path.stat().st_size / (1024 * 1024)
        output_files["aiperf_multi_turn"] = str(jsonl_path)
        print(f"aiperf multi_turn JSONL written to: {jsonl_path} ({mb:.2f} MB)")

    if fmt in ("all", "mooncake"):
        entries = convert_to_aiperf_mooncake(conversations)
        mooncake_path = output_dir / f"{file_base}_mooncake.jsonl"
        with open(mooncake_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        mb = mooncake_path.stat().st_size / (1024 * 1024)
        output_files["mooncake_trace"] = str(mooncake_path)
        print(f"aiperf mooncake_trace JSONL written to: {mooncake_path} ({mb:.2f} MB)")

    if not args.no_profile:
        manifest = build_manifest(
            df=df,
            config=config,
            dataset_type="pdf",
            seed=seed,
            output_files=output_files,
            descriptive_name=descriptive_name,
        )
        manifest_path = save_manifest(manifest, output_dir, file_base)
        output_files["manifest"] = str(manifest_path)
        print(f"\nDataset manifest written to: {manifest_path}")
        print_profile_summary(manifest)


if __name__ == "__main__":
    main()
