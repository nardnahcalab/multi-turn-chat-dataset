#!/usr/bin/env python3
"""
Synthetic multi-turn deep reasoning conversation generator for inference benchmarking.

Generates conversations that prompt models to perform extended chain-of-thought
reasoning across mathematical proofs, logic puzzles, algorithmic analysis,
scientific reasoning, philosophical arguments, game theory, causal reasoning,
and puzzle solving.

Designed to stress-test reasoning-heavy inference workloads where long output
sequences and deep thinking are required.

Usage:
    python generate.py                     # uses default config.yaml
    python generate.py --config my.yaml    # custom config
    python generate.py --num 1000          # override conversation count
"""

import argparse
import json
import random
import re
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# ---------------------------------------------------------------------------
# Topic-specific conversation templates
# ---------------------------------------------------------------------------

TOPIC_TEMPLATES = {
    "mathematical_proofs": {
        "openers": [
            "Prove that for all positive integers n, the sum 1 + 2 + 3 + ... + n equals n(n+1)/2. I want a rigorous proof, not just verification for a few cases.",
            "Show that the square root of {prime_number} is irrational. Walk me through every step of the proof.",
            "Prove by induction that {induction_statement}. Make sure to clearly state the base case and inductive step.",
            "I'm trying to prove that there are infinitely many prime numbers. Can you reconstruct Euclid's proof and explain why each step is necessary?",
            "Prove that if a and b are integers such that a^2 + b^2 is divisible by 3, then both a and b must be divisible by 3.",
            "Derive the closed-form solution for the Fibonacci recurrence F(n) = F(n-1) + F(n-2) using the characteristic equation method. Show all the algebra.",
            "Prove that the function f(x) = {function_expr} is continuous at x = {x_value} using the epsilon-delta definition of continuity.",
            "Show that every {algebraic_structure} with {property} must also satisfy {derived_property}. I need a complete proof.",
            "Prove the Cauchy-Schwarz inequality for vectors in R^n. Then explain geometrically why it must be true.",
            "I need to prove that {combinatorial_identity}. Can you show me both an algebraic proof and a combinatorial (counting) argument?",
        ],
        "followups": [
            "Wait, in step {step_number} you assumed {assumption}. Can you justify why that's valid?",
            "What happens if we relax the condition that {condition}? Does the result still hold?",
            "Can you prove it a different way? I want to see an alternative approach using {alt_method}.",
            "I follow the logic up to the point where you said {claim}. Can you expand on why that follows from the previous steps?",
            "What's the converse of this theorem? Is the converse also true? Prove or disprove it.",
            "How does this generalize to {generalization}? Does the same proof technique work?",
            "You used {technique} in the proof. When does this technique fail, and what are its limitations?",
            "Can you formalize the proof more rigorously? I want it in a form that could appear in a graduate textbook.",
            "What are the necessary and sufficient conditions for this result? You've shown sufficiency — what about necessity?",
            "I tried extending your proof to {extension_case} and got stuck at {stuck_point}. Where does the argument break down?",
            "Can you verify this with a concrete numerical example to build my intuition before we move on?",
            "What's the weakest hypothesis under which this theorem holds?",
        ],
        "responses": [
            "Let me construct this proof step by step.\n\n**Claim:** {claim}\n\n**Proof:**\n\n*Base case:* {base_case}\n\n*Inductive hypothesis:* Assume the statement holds for some arbitrary k >= {base_value}. That is, assume {inductive_hypothesis}.\n\n*Inductive step:* We need to show the statement holds for k+1.\n\n{inductive_step}\n\nBy the principle of mathematical induction, the statement holds for all {quantifier}. **QED**\n\nThe key insight here is {key_insight}. This is a {proof_type} proof, which works because {explanation}.",
            "This is a classic result. The standard proof proceeds by contradiction.\n\n**Proof:**\n\nSuppose, for the sake of contradiction, that {negation}.\n\nThen we can write {consequence1}.\n\nFrom this it follows that {consequence2}.\n\nBut {consequence3}, which contradicts our assumption that {original_assumption}.\n\nTherefore, our supposition was false, and {conclusion}. **QED**\n\nNote that the contradiction arises because {contradiction_source}. This is significant because {significance}.",
            "Here's a direct proof:\n\n**Setup:** Let {variables}.\n\n**Step 1:** {proof_step1}\n\n**Step 2:** From Step 1, we have {intermediate1}. Applying {technique}, we get {intermediate2}.\n\n**Step 3:** {proof_step3}\n\n**Step 4:** Combining Steps 2 and 3: {combination}\n\nTherefore, {conclusion}. **QED**\n\nThis proof technique is called {technique_name}. It works here because {technique_reason}. A common mistake is to {common_mistake}, which would invalidate the argument because {mistake_reason}.",
            "Let me prove this using {method}.\n\n**Definitions:** Recall that {definition}.\n\n**Lemma:** {lemma_statement}\n\n*Proof of Lemma:* {lemma_proof}\n\n**Main Theorem:** Using the lemma, we can now prove the main result.\n\n{main_proof}\n\nThe lemma was necessary because {lemma_necessity}. Without it, we cannot bridge the gap between {gap_from} and {gap_to}.",
        ],
        "fill_values": {
            "prime_number": ["2", "3", "5", "7", "11", "13"],
            "induction_statement": [
                "1^2 + 2^2 + ... + n^2 = n(n+1)(2n+1)/6",
                "2^n > n^2 for all n >= 5",
                "n! > 2^n for all n >= 4",
                "the number of subsets of an n-element set is 2^n",
                "a set with n elements has n(n-1)/2 two-element subsets",
            ],
            "function_expr": ["x^2 + 3x - 1", "sin(x)/x for x != 0", "(x^3 - 8)/(x - 2) for x != 2"],
            "x_value": ["0", "1", "2", "pi"],
            "algebraic_structure": ["group", "ring", "field", "vector space"],
            "property": ["finite order", "no zero divisors", "every element is invertible", "dimension n"],
            "derived_property": ["commutativity", "the cancellation property", "unique factorization", "a basis exists"],
            "combinatorial_identity": [
                "C(n,k) = C(n-1,k-1) + C(n-1,k)",
                "the sum of C(n,k) for k=0 to n equals 2^n",
                "C(2n,n) = sum of C(n,k)^2 for k=0 to n",
            ],
            "alt_method": ["contradiction", "contrapositive", "double counting", "generating functions", "direct construction"],
            "technique": ["proof by contradiction", "strong induction", "the pigeonhole principle", "a counting argument"],
            "generalization": ["higher dimensions", "arbitrary fields", "infinite sets", "non-commutative structures"],
            "step_number": ["3", "4", "5", "2"],
        },
    },

    "logic_and_deduction": {
        "openers": [
            "There are {num_people} people: {people_names}. Each one is either a knight (always tells the truth) or a knave (always lies). {person_a} says: \"{statement_a}\". {person_b} says: \"{statement_b}\". Determine who is a knight and who is a knave. Show your complete reasoning.",
            "Evaluate the following argument for validity:\n\nPremise 1: {premise1}\nPremise 2: {premise2}\nConclusion: {conclusion}\n\nIs this a valid deductive argument? Identify the logical form and explain your reasoning.",
            "Consider the following logical puzzle: {logic_puzzle}. Solve it step by step, showing your chain of deductions.",
            "Translate the following English statements into propositional logic, then determine if the argument is valid using a truth table or proof:\n\n1. {statement1}\n2. {statement2}\n3. Therefore: {statement3}",
            "In a room there are {num_people} people. You know the following facts:\n{fact_list}\nDetermine {query}. Show every step of your deduction.",
            "A detective has {num_suspects} suspects: {suspects}. The clues are:\n{clue_list}\nWho committed the crime? Walk through your elimination process.",
            "Prove or disprove: \"{logical_claim}\". Use formal logic to justify your answer.",
            "Consider the paradox: \"{paradox}\". Analyze this carefully. Where does the reasoning go wrong, or is it a genuine paradox? Explain.",
        ],
        "followups": [
            "What if {person_a} had instead said \"{alt_statement}\"? Would your conclusion change?",
            "Can you construct a truth table showing all possible cases? I want to see every row.",
            "You said {claim} — but couldn't {counter_scenario} also be consistent with the given information?",
            "Is there a shorter deduction path? Your reasoning is correct but seems longer than necessary.",
            "What logical fallacy would someone commit if they concluded {wrong_conclusion} from these premises?",
            "Now add a third premise: \"{new_premise}\". Does this change the validity of the argument?",
            "Can you express your solution using formal notation (quantifiers, connectives) so I can verify it mechanically?",
            "What's the minimum number of clues needed to uniquely determine the answer?",
            "Suppose we remove the constraint that {removed_constraint}. How many solutions exist then?",
            "What meta-logical principle are you relying on in step {step_number}?",
            "Can you verify your answer by assuming the opposite and deriving a contradiction?",
            "Is this problem decidable in general? What's its computational complexity?",
        ],
        "responses": [
            "Let me work through this systematically.\n\n**Given information:**\n{given_info}\n\n**Case analysis:**\n\n*Case 1:* Suppose {case1_assumption}.\n{case1_reasoning}\nThis leads to {case1_result}.\n\n*Case 2:* Suppose {case2_assumption}.\n{case2_reasoning}\nThis leads to {case2_result}.\n\n**Conclusion:** {case_conclusion}\n\nThe reasoning is valid because {validity_reason}. We've exhaustively checked all possibilities, which is what makes deductive reasoning so powerful — we can be *certain* of the answer, not merely confident.",
            "Let me formalize this argument.\n\n**Symbolization:**\n{symbolization}\n\n**Logical form:**\n{logical_form}\n\n**Evaluation:**\n{evaluation}\n\nThis argument is **{validity}** because {validity_explanation}.\n\n{additional_note}",
            "I'll solve this by process of elimination, tracking what we can deduce at each step.\n\n**Step 1:** From clue {clue_ref1}: {deduction1}\n\n**Step 2:** From clue {clue_ref2} combined with Step 1: {deduction2}\n\n**Step 3:** From clue {clue_ref3} and what we now know: {deduction3}\n\n**Step 4:** The only remaining possibility: {deduction4}\n\n**Verification:** Let me check that our solution satisfies all the original clues:\n{verification}\n\nAll clues check out. The answer is **{answer}**.",
        ],
        "fill_values": {
            "num_people": ["3", "4", "5"],
            "people_names": ["Alice, Bob, and Carol", "Alice, Bob, Carol, and Dave", "Pat, Quinn, and Riley"],
            "person_a": ["Alice", "Pat", "The first person"],
            "person_b": ["Bob", "Quinn", "The second person"],
            "statement_a": [
                "Bob is a knave",
                "At least one of us is a knight",
                "Carol and I are the same type",
                "Exactly two of us are knaves",
            ],
            "statement_b": [
                "Alice and I are not both knights",
                "If Alice is a knight, then Carol is a knave",
                "Alice is lying",
                "We are all knaves",
            ],
            "premise1": [
                "All humans are mortal",
                "If it rains, the ground gets wet",
                "Every prime greater than 2 is odd",
                "If a number is divisible by 6, it is divisible by 2",
            ],
            "premise2": [
                "Socrates is human",
                "The ground is not wet",
                "17 is greater than 2",
                "The number n is divisible by 6",
            ],
            "conclusion": [
                "Socrates is mortal",
                "It did not rain",
                "17 is odd",
                "n is divisible by 2",
            ],
            "num_suspects": ["4", "5", "3"],
            "suspects": ["Mr. Green, Mrs. White, Colonel Mustard, and Professor Plum",
                        "Adams, Baker, Clark, Davis, and Evans"],
            "logical_claim": [
                "If P implies Q, and Q implies R, then P implies R",
                "If not-P implies a contradiction, then P must be true",
                "It is possible for an argument to have true premises, valid form, and a false conclusion",
            ],
            "paradox": [
                "This statement is false",
                "The barber shaves all those and only those who do not shave themselves. Who shaves the barber?",
                "If God is omnipotent, can God create a stone so heavy that even God cannot lift it?",
            ],
        },
    },

    "algorithmic_analysis": {
        "openers": [
            "Design an efficient algorithm to solve the following problem: {algo_problem}. Analyze its time and space complexity, and prove that your solution is correct.",
            "I have a function that runs in O({current_complexity}) for {problem_description}. Can you help me find an O({target_complexity}) solution? Prove that the improved complexity is achievable.",
            "Prove that the problem of {hard_problem} is NP-hard by reducing from {known_np_problem}. Show the complete reduction.",
            "Analyze the following recurrence relation: T(n) = {recurrence}. Find the closed-form solution and prove it using the {analysis_method}.",
            "Compare the following approaches for {problem_description}:\n\nApproach A: {approach_a}\nApproach B: {approach_b}\n\nWhich is better and under what conditions? Provide a rigorous analysis.",
            "Given a {data_structure} with n elements, prove that {operation} requires at least {lower_bound} time in the worst case. I want an information-theoretic lower bound argument.",
            "Design a {paradigm} algorithm for the {problem_name} problem. Prove its correctness using a loop invariant (or structural induction for recursive solutions).",
            "Can you walk me through the amortized analysis of {data_structure_operation}? I want to understand why the average cost per operation is {amortized_cost} even though individual operations can cost {worst_case_cost}.",
        ],
        "followups": [
            "What if the input isn't sorted? How does that change the complexity?",
            "Can you prove a matching lower bound? I want to know if the algorithm is optimal.",
            "What about space complexity? Can we do better with an in-place approach?",
            "Your invariant says {invariant}. How do you prove it holds after the {operation_step} step?",
            "What happens in the average case? Is the worst case actually common in practice?",
            "How would this work on a distributed system with {num_nodes} nodes?",
            "Can you trace through the algorithm on the input {trace_input} so I can verify my understanding?",
            "What's the best known algorithm for this problem? Is ours close to optimal?",
            "How does the constant factor compare between approaches? Big-O hides that.",
            "What if we allow randomization? Can we get a better expected time?",
            "Can this be parallelized? What's the work-span tradeoff?",
            "Your proof uses {technique} — can you explain why this technique is appropriate here?",
        ],
        "responses": [
            "Here's an efficient algorithm with a rigorous analysis.\n\n**Algorithm:**\n```\n{pseudocode}\n```\n\n**Correctness Proof:**\n\n*Loop invariant:* {invariant}\n\n*Initialization:* {init_proof}\n\n*Maintenance:* {maintenance_proof}\n\n*Termination:* {termination_proof}\n\n**Complexity Analysis:**\n\n*Time:* {time_analysis}\n\n*Space:* {space_analysis}\n\nThe key insight is {key_insight}. This is better than the naive approach because {improvement_reason}.",
            "Let me analyze this recurrence systematically.\n\n**Recurrence:** T(n) = {recurrence_statement}\n\n**Method:** {method_name}\n\n**Step 1:** {solve_step1}\n\n**Step 2:** {solve_step2}\n\n**Step 3:** {solve_step3}\n\n**Result:** T(n) = {closed_form}\n\n**Verification:** Let me verify with small values:\n{verification}\n\nThis tells us the algorithm runs in {complexity_class} time. Intuitively, this makes sense because {intuition}.",
            "I'll prove this is NP-hard by reduction.\n\n**Reduction from {source_problem} to {target_problem}:**\n\n**Construction:** Given an instance I of {source_problem}, construct an instance I' of {target_problem} as follows:\n{construction}\n\n**Forward direction:** If I is a YES-instance, then {forward_proof}\n\n**Backward direction:** If I' is a YES-instance, then {backward_proof}\n\n**Polynomial time:** The construction runs in {construction_time} because {time_justification}.\n\nSince {source_problem} is NP-hard, {target_problem} is also NP-hard. **QED**",
            "Let me compare both approaches rigorously.\n\n**Approach A ({approach_a_name}):**\n- Time: {time_a}\n- Space: {space_a}\n- Best when: {best_a}\n\n**Approach B ({approach_b_name}):**\n- Time: {time_b}\n- Space: {space_b}\n- Best when: {best_b}\n\n**Crossover point:** {crossover}\n\n**Recommendation:** {recommendation}\n\nThe analysis shows {conclusion}. In practice, the constant factors matter: {practical_note}.",
        ],
        "fill_values": {
            "algo_problem": [
                "finding the k-th smallest element in an unsorted array",
                "merging k sorted linked lists into one sorted list",
                "finding the longest increasing subsequence",
                "computing the edit distance between two strings",
                "finding the maximum flow in a network",
            ],
            "current_complexity": ["n^2", "n^2 log n", "2^n", "n^3"],
            "target_complexity": ["n log n", "n log k", "n*k", "n^2"],
            "problem_description": [
                "sorting n elements", "searching in a graph with V vertices and E edges",
                "string matching with pattern length m in text length n",
                "matrix multiplication of two n x n matrices",
            ],
            "hard_problem": [
                "finding the minimum vertex cover", "3-coloring a graph",
                "the traveling salesman decision problem", "subset sum",
            ],
            "known_np_problem": ["3-SAT", "VERTEX-COVER", "INDEPENDENT-SET", "CLIQUE"],
            "recurrence": [
                "2T(n/2) + O(n)", "T(n/2) + O(1)", "2T(n/2) + O(n log n)",
                "T(n-1) + T(n-2) + O(1)", "3T(n/4) + O(n)",
            ],
            "analysis_method": ["Master Theorem", "substitution method", "recursion tree method"],
            "paradigm": ["divide and conquer", "dynamic programming", "greedy", "backtracking"],
            "data_structure": ["binary search tree", "hash table", "heap", "balanced BST"],
            "operation": ["comparison-based sorting", "searching", "finding the median"],
            "lower_bound": ["Omega(n log n)", "Omega(log n)", "Omega(n)"],
            "problem_name": ["interval scheduling", "knapsack", "matrix chain multiplication", "coin change"],
        },
    },

    "scientific_reasoning": {
        "openers": [
            "A researcher observes that {observation}. They hypothesize that {hypothesis}. Design an experiment to test this hypothesis. What are the key variables, controls, and potential confounders?",
            "Two studies report conflicting results: Study A finds {finding_a}, while Study B finds {finding_b}. Both are published in reputable journals. How should we reconcile these results? Walk through your reasoning.",
            "Consider the claim: \"{scientific_claim}\". What evidence would you need to see to accept this claim? What evidence would falsify it? How strong must the evidence be?",
            "A new drug shows {drug_result} in a sample of {sample_size} patients. The p-value is {p_value}. Should we conclude the drug is effective? Analyze this critically.",
            "Explain the reasoning behind the scientific consensus on {scientific_topic}. What are the strongest pieces of evidence? What would it take to overturn this consensus?",
            "Propose a mechanism that could explain why {phenomenon}. Then describe how you would test each component of your proposed mechanism.",
            "A dataset shows a strong correlation ({r_value}) between {variable_a} and {variable_b}. A journalist writes \"{headline}\". Critique this conclusion. What alternative explanations exist?",
            "Design a thought experiment to resolve the question: {thought_experiment_question}. What assumptions are you making, and how do they affect your conclusions?",
        ],
        "followups": [
            "You mentioned {confounder} as a potential confounder. How exactly would it bias the results?",
            "What sample size would be needed to detect an effect of this magnitude with {power}% power?",
            "Could this be explained by {alt_explanation} instead? How would you distinguish between the two hypotheses?",
            "What's the difference between statistical significance and practical significance here?",
            "If we repeated this experiment 100 times, how often would we expect to see a false positive?",
            "You assumed {assumption}. How sensitive are the conclusions to this assumption?",
            "Can you draw a causal diagram (DAG) for this situation? I want to see all the relationships.",
            "What Bayesian prior should we assign to this hypothesis before seeing the data?",
            "The study used {methodology}. What are the specific limitations of this approach?",
            "How would you rule out the placebo effect / Hawthorne effect / regression to the mean?",
            "What would a preregistered version of this experiment look like?",
            "Is this result likely to replicate? What factors affect replicability?",
        ],
        "responses": [
            "Let me design a rigorous experiment.\n\n**Hypothesis:** {hypothesis}\n\n**Independent variable:** {iv}\n**Dependent variable:** {dv}\n**Control variables:** {controls}\n\n**Design:** {design_type}\n\n**Procedure:**\n1. {proc_step1}\n2. {proc_step2}\n3. {proc_step3}\n\n**Potential confounders and how to address them:**\n{confounders}\n\n**Statistical analysis plan:** {stat_plan}\n\n**Power analysis:** {power_analysis}\n\nThe key strength of this design is {strength}. The main limitation is {limitation}, which we can partially mitigate by {mitigation}.",
            "This is a great example of why we need to reason carefully about causation.\n\n**The correlation:** {correlation}\n\n**Why it might be causal:** {causal_argument}\n\n**Why it might NOT be causal:**\n1. **Reverse causation:** {reverse}\n2. **Confounding:** {confounding}\n3. **Selection bias:** {selection_bias}\n4. **Measurement artifact:** {artifact}\n\n**To establish causation, we would need:**\n{causation_criteria}\n\nThe fundamental issue is that {fundamental_issue}. This is exactly why {methodological_principle}.",
            "Let me evaluate the evidence systematically.\n\n**Claim:** {claim}\n\n**Evidence FOR:**\n{evidence_for}\n\n**Evidence AGAINST:**\n{evidence_against}\n\n**Quality assessment:**\n{quality_assessment}\n\n**My assessment:** {assessment}\n\nThe confidence level depends critically on {critical_factor}. If we apply {framework}, the weight of evidence suggests {conclusion}. However, I'd rate my confidence at about {confidence_level} because {confidence_reason}.",
        ],
        "fill_values": {
            "observation": [
                "cities with more ice cream sales have higher crime rates",
                "students who eat breakfast score 12% higher on standardized tests",
                "patients given the new treatment recover 3 days faster on average",
                "countries with more cell phone towers have lower infant mortality",
            ],
            "hypothesis": [
                "ice cream consumption causes criminal behavior",
                "eating breakfast improves cognitive function in students",
                "the treatment accelerates immune system response",
                "cell phone access improves health outcomes through information access",
            ],
            "finding_a": [
                "a significant positive effect (d=0.4, p<0.01, n=200)",
                "no statistically significant difference (p=0.3, n=500)",
            ],
            "finding_b": [
                "no significant effect (d=0.05, p=0.6, n=150)",
                "a significant negative effect (d=-0.3, p<0.05, n=80)",
            ],
            "scientific_claim": [
                "Moderate red wine consumption reduces the risk of heart disease",
                "Humans only use 10% of their brain",
                "The Flynn Effect shows that human intelligence is increasing over time",
                "Sleep deprivation causes weight gain through hormonal disruption",
            ],
            "drug_result": ["a 15% reduction in symptom severity", "a 30% improvement in recovery time"],
            "sample_size": ["50", "200", "1000"],
            "p_value": ["0.04", "0.001", "0.049"],
            "scientific_topic": ["evolution by natural selection", "anthropogenic climate change", "germ theory of disease"],
            "variable_a": ["screen time", "chocolate consumption", "shoe size"],
            "variable_b": ["depression scores", "Nobel Prize counts by country", "reading ability in children"],
            "r_value": ["r=0.72", "r=0.85", "r=0.68"],
            "headline": [
                "Screen Time Causes Depression, Study Finds",
                "Eating Chocolate Makes Countries Smarter",
                "Bigger Feet Linked to Better Reading",
            ],
            "methodology": ["observational cohort study", "randomized controlled trial", "meta-analysis", "case-control study"],
            "power": ["80", "90", "95"],
        },
    },

    "philosophical_arguments": {
        "openers": [
            "Consider the trolley problem: A runaway trolley is heading toward {num_people} people. You can divert it to a side track where it will kill {num_fewer} person. Should you divert it? Analyze this from {num_frameworks} different ethical frameworks.",
            "Is it possible to know anything with absolute certainty? Construct and evaluate Descartes' argument from radical doubt. Where does it succeed and where does it fail?",
            "Present the strongest version of {philosophical_argument}. Then present the strongest objection to it. Which side do you find more compelling, and why?",
            "Consider the thought experiment: {thought_experiment}. What does this tell us about {philosophical_topic}? Explore at least three different interpretations.",
            "Are {abstract_concept_a} and {abstract_concept_b} fundamentally the same thing, or are they genuinely distinct? Construct arguments for both positions.",
            "A friend argues: \"{informal_argument}\". Reconstruct this argument in its strongest form, identify hidden premises, then evaluate it. Is it sound?",
            "Is {ethical_action} morally permissible? Analyze using utilitarian, deontological, and virtue ethics frameworks. Do they agree?",
            "What is the relationship between {concept_a} and {concept_b}? Is one more fundamental than the other? Defend your position with rigorous arguments.",
        ],
        "followups": [
            "You're assuming {hidden_assumption}. Can you justify that assumption, or does the argument work without it?",
            "A {philosopher} would object that {objection}. How would you respond?",
            "What if we modify the scenario so that {modification}? Does your analysis change?",
            "You said {claim}. But isn't that just {fallacy_name}? Explain why it's not (or admit if it is).",
            "Can you steelman the opposing position? I want the strongest possible version of the view you disagree with.",
            "How does this connect to {related_topic}? I see a tension between what you said here and the standard view on that.",
            "What real-world implications follow from your conclusion? Give me a concrete example.",
            "Is this argument culturally dependent, or does it hold universally? What would a cross-cultural analysis show?",
            "Where does intuition end and rigorous argument begin in your analysis?",
            "If you had to assign a probability to your conclusion being correct, what would it be, and why?",
            "How would an {alt_tradition} philosopher approach this same question?",
        ],
        "responses": [
            "This is one of the deepest questions in philosophy. Let me analyze it carefully.\n\n**The core tension:** {core_tension}\n\n**Position A: {position_a_name}**\n{position_a_argument}\n\n**Position B: {position_b_name}**\n{position_b_argument}\n\n**Evaluation:**\n\nPosition A is stronger in that {a_strength}, but it faces the serious objection that {a_weakness}.\n\nPosition B avoids this problem, but {b_weakness}.\n\n**My assessment:** I find {preferred_position} more compelling because {preference_reason}. However, this conclusion depends critically on {critical_assumption}, which is itself debatable.",
            "Let me reconstruct the argument formally.\n\n**Premise 1:** {premise1}\n**Premise 2:** {premise2}\n**Premise 3:** {premise3}\n**Conclusion:** {conclusion}\n\n**Evaluation of each premise:**\n\n*Premise 1:* {eval1}\n*Premise 2:* {eval2}\n*Premise 3:* {eval3}\n\n**Validity:** The argument is {validity} — the conclusion {validity_detail}.\n\n**Soundness:** Even if valid, the argument is {soundness} because {soundness_reason}.\n\n{historical_note}",
            "Let me apply three ethical frameworks.\n\n**1. Utilitarian analysis:**\n{utilitarian_analysis}\n*Verdict:* {utilitarian_verdict}\n\n**2. Deontological analysis (Kantian):**\n{deontological_analysis}\n*Verdict:* {deontological_verdict}\n\n**3. Virtue ethics analysis:**\n{virtue_analysis}\n*Verdict:* {virtue_verdict}\n\n**Synthesis:**\nInterestingly, {synthesis}. The frameworks {agreement_status}, which tells us {meta_insight}.\n\nThe fundamental disagreement comes down to {fundamental_disagreement}. This is not something that can be resolved by additional facts — it's a genuinely philosophical question about {deep_question}.",
        ],
        "fill_values": {
            "num_people": ["5", "10", "100"],
            "num_fewer": ["1", "2"],
            "num_frameworks": ["3", "4"],
            "philosophical_argument": [
                "the cosmological argument for God's existence",
                "the argument from evil against God's existence",
                "the simulation argument",
                "the Chinese Room argument against strong AI",
                "the hard problem of consciousness",
            ],
            "thought_experiment": [
                "Mary the color scientist who has never seen color",
                "the ship of Theseus where every plank is gradually replaced",
                "a perfect teleporter that destroys and recreates you atom by atom",
                "a brain in a vat receiving perfect simulated inputs",
                "Searle's Chinese Room",
            ],
            "philosophical_topic": ["consciousness", "personal identity", "the nature of knowledge", "free will"],
            "abstract_concept_a": ["free will", "consciousness", "justice", "truth"],
            "abstract_concept_b": ["determinism", "intelligence", "fairness", "belief"],
            "ethical_action": [
                "lying to spare someone's feelings",
                "breaking a promise to prevent a greater harm",
                "sacrificing one person to save five",
                "using AI to make life-or-death medical decisions",
            ],
            "philosopher": ["Kant", "Hume", "Nietzsche", "Singer", "Rawls", "Nozick"],
            "concept_a": ["knowledge", "mind", "causation", "morality"],
            "concept_b": ["justified true belief", "brain states", "regularity", "evolutionary advantage"],
            "informal_argument": [
                "If we can't prove free will exists, we should just act as if it does",
                "Morality is just whatever society agrees on",
                "Science can never answer questions about meaning and purpose",
                "The fact that we can imagine zombies proves consciousness is non-physical",
            ],
            "alt_tradition": ["Buddhist", "Confucian", "existentialist", "pragmatist", "feminist"],
        },
    },

    "game_theory_and_strategy": {
        "openers": [
            "Two firms are deciding simultaneously whether to set high or low prices. The payoff matrix is:\n\n|  | Firm B: High | Firm B: Low |\n|---|---|---|\n| Firm A: High | ({payoff_hh_a}, {payoff_hh_b}) | ({payoff_hl_a}, {payoff_hl_b}) |\n| Firm A: Low | ({payoff_lh_a}, {payoff_lh_b}) | ({payoff_ll_a}, {payoff_ll_b}) |\n\nFind all Nash equilibria. Is this a prisoner's dilemma? Explain your reasoning.",
            "Design an auction mechanism for selling {auction_item}. The {num_bidders} bidders have private valuations. What mechanism maximizes {objective}? Prove that your mechanism is {mechanism_property}.",
            "In a game of {game_name}, {game_setup}. Determine the optimal strategy using backward induction. Show the complete game tree analysis.",
            "Consider a repeated prisoner's dilemma played {num_rounds}. Analyze when cooperation can be sustained as an equilibrium. What role does the discount factor play?",
            "A decision maker faces the following uncertain scenario: {decision_scenario}. Use {decision_framework} to determine the optimal choice. Show your calculations.",
            "Analyze the following mechanism: {mechanism_description}. Is it incentive-compatible? Is it individually rational? Prove your answers.",
            "There are {num_players} players who must divide {resource} among themselves. Design a division mechanism that is {fairness_criterion}. Prove it satisfies the criterion.",
        ],
        "followups": [
            "What if the game is repeated infinitely? How does that change the equilibrium analysis?",
            "Is the Nash equilibrium you found also Pareto efficient? If not, is there a Pareto-improving outcome?",
            "What happens if one player can commit to a strategy before the other moves? Model this as a Stackelberg game.",
            "How robust is this equilibrium? What if players are boundedly rational?",
            "Can you find the mixed strategy Nash equilibrium? What are the exact probabilities?",
            "What's the price of anarchy in this game? How bad is the Nash equilibrium compared to the social optimum?",
            "What if players can communicate but can't make binding agreements? Does that change anything?",
            "How would you compute this equilibrium for the case of {num_players} players instead of 2?",
            "Is there a dominant strategy for either player? What about iterated elimination of dominated strategies?",
            "What real-world scenario does this game model? Give me a concrete example.",
            "Can you show that no mechanism can simultaneously satisfy {property_a}, {property_b}, and {property_c}? (An impossibility result.)",
        ],
        "responses": [
            "Let me find the Nash equilibria systematically.\n\n**Step 1: Identify best responses.**\n\nFor Firm A:\n{best_response_a}\n\nFor Firm B:\n{best_response_b}\n\n**Step 2: Find mutual best responses.**\n{mutual_best}\n\n**Nash Equilibrium/Equilibria:** {equilibria}\n\n**Is it a prisoner's dilemma?** {pd_analysis}\n\n{efficiency_analysis}\n\nThe intuition is {intuition}. This has important real-world implications for {real_world}.",
            "I'll use backward induction to solve this extensive-form game.\n\n**Game tree:**\n{game_tree}\n\n**Working backwards:**\n\n*At the final decision node:* {final_node}\n*At the second-to-last node:* {penultimate_node}\n*At the first decision node:* {first_node}\n\n**Subgame-perfect equilibrium:** {spe}\n\n**Path of play:** {path}\n\n**Payoffs:** {payoffs}\n\nThe key strategic insight is {strategic_insight}. Notice that {observation}, which illustrates {principle}.",
            "Let me analyze this mechanism rigorously.\n\n**The mechanism:** {mechanism_summary}\n\n**Incentive compatibility:**\n\n*Claim:* Truth-telling is a dominant strategy.\n\n*Proof:* Consider agent i with true value v_i. Suppose they report v' != v_i.\n\n{ic_proof}\n\nTherefore, no agent can gain by misreporting. **QED**\n\n**Individual rationality:**\n\n{ir_proof}\n\n**Revenue/efficiency:**\n{revenue_analysis}\n\nThis mechanism is related to the {related_mechanism} and illustrates the principle that {mechanism_principle}.",
        ],
        "fill_values": {
            "payoff_hh_a": ["3", "5", "4"], "payoff_hh_b": ["3", "5", "4"],
            "payoff_hl_a": ["0", "1", "0"], "payoff_hl_b": ["5", "8", "6"],
            "payoff_lh_a": ["5", "8", "6"], "payoff_lh_b": ["0", "1", "0"],
            "payoff_ll_a": ["1", "2", "2"], "payoff_ll_b": ["1", "2", "2"],
            "auction_item": ["a single indivisible good", "multiple identical items", "spectrum licenses"],
            "num_bidders": ["3", "5", "n"],
            "objective": ["revenue", "social welfare", "efficiency"],
            "mechanism_property": ["incentive-compatible", "individually rational", "strategy-proof"],
            "game_name": ["ultimatum", "centipede", "sequential bargaining", "entry deterrence"],
            "num_rounds": ["finitely many times", "100 rounds", "an unknown number of rounds"],
            "num_players": ["3", "4", "n"],
            "resource": ["$100", "a divisible resource", "10 identical items"],
            "fairness_criterion": ["envy-free", "proportional", "equitable"],
            "decision_framework": ["expected utility theory", "minimax regret", "maximin"],
            "property_a": ["strategy-proofness", "efficiency"],
            "property_b": ["budget balance", "fairness"],
            "property_c": ["individual rationality", "non-dictatorship"],
        },
    },

    "causal_and_counterfactual": {
        "openers": [
            "A company ran an A/B test: group A ({group_a_size} users) saw {treatment_a}, group B ({group_b_size} users) saw {treatment_b}. Group A had a {metric} of {result_a}, Group B had {result_b}. Can we conclude the treatment caused the difference? Analyze thoroughly.",
            "Draw a causal DAG for the following scenario: {dag_scenario}. Identify all confounders, mediators, and colliders. Then determine which variables must be controlled for to estimate the causal effect of {cause} on {effect}.",
            "Consider the counterfactual: \"If {counterfactual_event}, then {counterfactual_outcome}.\" Evaluate whether this counterfactual is true. What framework should we use, and what assumptions are needed?",
            "A government claims that {policy} caused {outcome}. Critics argue it was actually due to {alt_cause}. Design an analysis strategy that could distinguish between these two explanations.",
            "Someone argues: \"{causal_claim}\". Identify at least {num_flaws} flaws in this causal reasoning. For each flaw, explain what additional evidence would be needed to fix it.",
            "We observe that {observation}. Using the potential outcomes framework, define the causal estimand of interest, identify the assumptions needed for estimation, and discuss whether those assumptions are plausible.",
            "Explain Simpson's paradox in the context of {simpson_context}. Show how the direction of an association can reverse when we condition on {conditioning_variable}. What's the correct causal interpretation?",
        ],
        "followups": [
            "You identified {variable} as a confounder. But what if it's actually a mediator? How would that change the analysis?",
            "Can you apply the do-calculus to this problem? I want to see if the causal effect is identifiable from observational data alone.",
            "What if there's an unmeasured confounder? How robust are your conclusions to that possibility?",
            "Walk me through the potential outcomes for a specific unit. What's the fundamental problem of causal inference here?",
            "How would an instrumental variable approach work here? What would be a valid instrument?",
            "Can you quantify the sensitivity to unmeasured confounding? Use the E-value or similar approach.",
            "What's the difference between the average treatment effect and the effect on the treated here? Which one is more relevant?",
            "If we could run a randomized experiment, what would it look like? What practical barriers prevent us from doing so?",
            "You used a {method} approach. What are the identifying assumptions, and which ones are most likely violated?",
            "How would a Granger causality test compare to the approach you've described? What are its limitations?",
            "Can you apply Pearl's front-door criterion or back-door criterion to this DAG?",
        ],
        "responses": [
            "Let me analyze the causal structure carefully.\n\n**Causal DAG:**\n{dag_description}\n\n**Key paths:**\n{paths}\n\n**Confounders (back-door paths):**\n{confounders}\n\n**Adjustment set:** To estimate the causal effect of {cause} on {effect}, we need to control for: {adjustment_set}\n\n**Why these variables:** {adjustment_justification}\n\n**Variables we should NOT control for:** {no_control} — because {no_control_reason}.\n\n**Identification result:** {identification}\n\nThis analysis assumes {assumptions}. The most questionable assumption is {weak_assumption} because {weak_reason}.",
            "Let me evaluate the A/B test results.\n\n**Observed difference:** {observed_diff}\n\n**Internal validity checks:**\n1. **Randomization:** {randomization_check}\n2. **Sample size adequacy:** {sample_check}\n3. **Attrition/dropout:** {attrition_check}\n4. **Compliance:** {compliance_check}\n\n**Threats to causal interpretation:**\n{threats}\n\n**Statistical analysis:**\n{statistical_analysis}\n\n**Conclusion:** {conclusion}\n\nThe strength of the A/B test is {strength}. However, even with randomization, {caveat}.",
            "This is a classic case of {causal_fallacy}.\n\n**The claim:** {original_claim}\n\n**Why it's flawed:**\n\n**Flaw 1: {flaw1_name}**\n{flaw1_detail}\n\n**Flaw 2: {flaw2_name}**\n{flaw2_detail}\n\n**Flaw 3: {flaw3_name}**\n{flaw3_detail}\n\n**What would fix the reasoning:**\n{fix_description}\n\n**The correct causal statement, based on available evidence, is:**\n{correct_statement}\n\nThis illustrates an important principle: {principle}.",
        ],
        "fill_values": {
            "group_a_size": ["5,000", "10,000", "50,000"],
            "group_b_size": ["5,000", "10,000", "50,000"],
            "treatment_a": ["the new landing page", "a 10% discount", "the redesigned checkout flow"],
            "treatment_b": ["the old landing page", "no discount", "the original checkout flow"],
            "metric": ["conversion rate", "average order value", "retention rate"],
            "result_a": ["4.2%", "$85", "68%"],
            "result_b": ["3.8%", "$79", "64%"],
            "dag_scenario": [
                "the relationship between education, income, and health outcomes",
                "the effect of a new drug on recovery, with treatment compliance and disease severity as factors",
                "the impact of social media use on teen mental health, with parental involvement as a potential moderator",
            ],
            "cause": ["education", "the treatment", "social media use"],
            "effect": ["health outcomes", "recovery time", "mental health"],
            "counterfactual_event": [
                "the Allies had not broken the Enigma code",
                "the internet had not been invented",
                "the patient had received the treatment instead of the placebo",
            ],
            "counterfactual_outcome": [
                "World War II would have lasted 2 more years",
                "scientific progress would be 20 years behind where it is now",
                "the patient would have recovered faster",
            ],
            "policy": ["the minimum wage increase", "the new education reform", "the lockdown measures"],
            "outcome": ["reduced poverty rates", "improved test scores", "reduced disease transmission"],
            "alt_cause": ["general economic growth", "demographic shifts", "seasonal patterns"],
            "causal_claim": [
                "Countries with more guns have more violence, so guns cause violence",
                "People who drink coffee live longer, so coffee extends lifespan",
                "Students who use tutors get better grades, proving that tutoring works",
            ],
            "num_flaws": ["3", "4"],
            "simpson_context": [
                "university admissions",
                "medical treatment effectiveness",
                "workplace discrimination analysis",
            ],
            "conditioning_variable": ["department", "disease severity", "job category"],
            "method": ["regression discontinuity", "difference-in-differences", "propensity score matching"],
        },
    },

    "puzzle_solving": {
        "openers": [
            "You have {num_items} {items} and a balance scale. One {item} is slightly {heavier_lighter} than the rest. Find the odd one out in the minimum number of weighings. Prove your solution is optimal.",
            "Solve this logic grid puzzle:\n{grid_puzzle}\nShow every step of your deduction.",
            "{num_people} people need to cross a bridge at night with one flashlight. The bridge holds at most {max_cross} people. Their crossing times are: {crossing_times}. What is the minimum total time? Prove it.",
            "A farmer needs to transport a {item_a}, a {item_b}, and a {item_c} across a river in a boat that can carry only the farmer and one item. {constraint}. How can the farmer get everything across safely? Is there more than one solution?",
            "You have two hourglasses: one measures {time_a} minutes and the other measures {time_b} minutes. How can you measure exactly {target_time} minutes? Prove it's possible (or impossible).",
            "Place {num_queens} queens on a {board_size}x{board_size} chessboard so that no two queens attack each other. How many distinct solutions exist? Show your reasoning.",
            "A census taker visits a house. The resident says: \"I have three children. The product of their ages is {product}. The sum of their ages is {sum_clue}.\" The census taker says \"I need more information.\" The resident replies: \"{additional_clue}.\" How old are the children? Show your reasoning.",
            "You are given {num_statements} statements. Exactly {num_true} of them are true:\n{statement_list}\nDetermine which statements are true. Prove your answer is the only consistent solution.",
        ],
        "followups": [
            "Can you prove that {num_steps} is the minimum number of steps? What's the information-theoretic lower bound?",
            "What if we change the constraint to {new_constraint}? Is the puzzle still solvable?",
            "How many distinct solutions exist? I want an exact count, not just one example.",
            "Can you generalize this to {generalization}? Is there a formula?",
            "I think there's a shorter solution. Can you prove no solution exists with fewer steps?",
            "What if {modification}? How does that change the answer?",
            "Is this puzzle NP-hard in general? What about for specific parameter values?",
            "What strategy would you use if you had to solve this under time pressure?",
            "Can you encode this as a constraint satisfaction problem and solve it systematically?",
            "What's the key insight that makes this puzzle tractable?",
            "Verify your solution by checking every constraint is satisfied.",
        ],
        "responses": [
            "Let me solve this systematically.\n\n**Analysis:** {analysis}\n\n**Strategy:**\n{strategy}\n\n**Solution:**\n\n*Step 1:* {step1}\n\n*Step 2:* {step2}\n\n*Step 3:* {step3}\n\n**Verification:** {verification}\n\n**Optimality proof:** {optimality}\n\nThe key insight is {key_insight}. This is related to {related_concept}.",
            "I'll use constraint propagation to solve this.\n\n**Initial constraints:**\n{constraints}\n\n**Deduction chain:**\n\n1. From constraint {c1}: {deduction1}\n2. From constraint {c2} + deduction 1: {deduction2}\n3. From constraint {c3} + deductions 1-2: {deduction3}\n4. From remaining constraints: {deduction4}\n\n**Solution:**\n{solution}\n\n**Uniqueness:** {uniqueness_proof}\n\n**Verification against all constraints:**\n{full_verification}",
            "Let me enumerate all possibilities and eliminate.\n\n**Total possibilities:** {total}\n\n**Applying constraint 1:** {elimination1} — reduces to {remaining1} possibilities.\n\n**Applying constraint 2:** {elimination2} — reduces to {remaining2}.\n\n**Applying constraint 3:** {elimination3} — reduces to {remaining3}.\n\n**Result:** {result}\n\nThe answer is **{answer}**.\n\nWhat makes this puzzle challenging is {difficulty_source}. The crucial step is {crucial_step}.",
        ],
        "fill_values": {
            "num_items": ["9", "12", "8", "27"],
            "items": ["coins", "balls", "weights"],
            "item": ["coin", "ball", "weight"],
            "heavier_lighter": ["heavier", "lighter"],
            "num_people": ["4", "5"],
            "max_cross": ["2"],
            "crossing_times": [
                "1, 2, 5, and 10 minutes",
                "1, 3, 6, 8, and 12 minutes",
            ],
            "item_a": ["fox", "wolf"],
            "item_b": ["chicken", "goat"],
            "item_c": ["grain", "cabbage"],
            "constraint": [
                "The fox will eat the chicken if left alone together, and the chicken will eat the grain",
                "The wolf will eat the goat if left alone, and the goat will eat the cabbage",
            ],
            "time_a": ["4", "7", "5"],
            "time_b": ["7", "11", "3"],
            "target_time": ["9", "15", "1"],
            "num_queens": ["8", "5", "6"],
            "board_size": ["8", "5", "6"],
            "product": ["36", "72", "120"],
            "sum_clue": ["the same as my house number", "13", "a prime number"],
            "additional_clue": ["My oldest child plays piano", "The oldest is a girl", "No twins"],
            "num_statements": ["5", "4", "6"],
            "num_true": ["2", "3"],
            "generalization": ["n items", "arbitrary board sizes", "k dimensions"],
            "new_constraint": ["the heavier item could be either heavier or lighter", "three people can cross at once"],
        },
    },
}

# ---------------------------------------------------------------------------
# Reasoning-specific content blocks for richer responses
# ---------------------------------------------------------------------------

PROOF_TECHNIQUES = [
    "proof by mathematical induction",
    "proof by contradiction (reductio ad absurdum)",
    "proof by contrapositive",
    "direct proof",
    "proof by exhaustion (case analysis)",
    "proof by construction",
    "proof by the pigeonhole principle",
    "proof by double counting",
    "proof using the well-ordering principle",
    "diagonalization argument",
]

REASONING_CHAINS = [
    "Let's think about this step by step. First, we need to identify what we know for certain. Then we can build up from there.",
    "The key observation is that this problem has a recursive structure. If we can solve the smaller case, we can extend it.",
    "Before jumping to a solution, let me consider what constraints are binding. Often the answer becomes clear once we find the bottleneck.",
    "This is deceptively simple at first glance, but there's a subtlety that most people miss. Let me be very careful about the assumptions.",
    "I'll approach this by first establishing an upper bound, then a lower bound, and then showing they match.",
    "Let me reason by analogy with a simpler version of the problem first, then generalize.",
    "The crucial step is to find the right invariant. Once we identify what doesn't change, everything else follows.",
    "Let me decompose this into sub-problems. Each one is tractable on its own, and their solutions combine to give us the full answer.",
]

MATHEMATICAL_CONCEPTS = [
    "This is a direct application of the fundamental theorem of arithmetic.",
    "Notice this is related to the Chinese Remainder Theorem, because the moduli are coprime.",
    "The pigeonhole principle gives us existence, but not construction — we know the object exists, but this proof doesn't tell us how to find it.",
    "By Bézout's identity, since gcd(a, b) = 1, there exist integers x, y such that ax + by = 1.",
    "This follows from the well-ordering principle: every non-empty subset of the natural numbers has a least element.",
    "The generating function approach transforms the recurrence into an algebraic equation, which we can solve using partial fractions.",
    "By the intermediate value theorem, since f is continuous and changes sign, it must have a root in the interval.",
    "This is an application of the inclusion-exclusion principle.",
]

LOGICAL_PRINCIPLES = [
    "This follows from modus ponens: if P implies Q, and P is true, then Q must be true.",
    "By modus tollens: if P implies Q, and Q is false, then P must be false.",
    "This is an application of proof by contradiction: we assume the negation and derive an impossibility.",
    "Note the distinction between 'necessary' and 'sufficient' conditions here — being necessary doesn't make it sufficient.",
    "De Morgan's laws tell us that NOT(A AND B) is equivalent to (NOT A) OR (NOT B).",
    "The law of excluded middle gives us that either P or NOT P must hold — there's no middle ground.",
    "Beware of the existential fallacy: just because all X are Y doesn't mean any X actually exist.",
    "This uses the transitive property of implication: if A implies B and B implies C, then A implies C.",
]

COMPLEXITY_RESULTS = [
    "This algorithm runs in O(n log n) time because each element participates in at most log n levels of recursion.",
    "The space complexity is O(n) due to the auxiliary array used in merging.",
    "By the Master Theorem (Case 2), T(n) = 2T(n/2) + O(n) solves to T(n) = O(n log n).",
    "The amortized cost per operation is O(1) by the aggregate method: n operations cost O(n) total.",
    "This is a decision problem in NP because a certificate (the solution itself) can be verified in polynomial time.",
    "The reduction preserves polynomial-time computability because each gadget construction takes O(1) time per variable/clause.",
    "The greedy algorithm achieves a 2-approximation ratio for this NP-hard problem.",
    "Dynamic programming reduces the exponential brute-force search to polynomial time by exploiting optimal substructure.",
]


# ---------------------------------------------------------------------------
# Conversation generator
# ---------------------------------------------------------------------------

class ReasoningConversationGenerator:
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.rng = random.Random(seed)
        self.topics = config["topics"]
        self.topic_weights = [t["weight"] for t in self.topics]

    def _fill_template(self, template: str, topic_name: str) -> str:
        """Fill placeholders in a template with random values."""
        fill = TOPIC_TEMPLATES[topic_name]["fill_values"]
        result = template
        max_iterations = 20
        iteration = 0
        while "{" in result and iteration < max_iterations:
            iteration += 1
            for key, values in fill.items():
                placeholder = "{" + key + "}"
                while placeholder in result:
                    result = result.replace(placeholder, self.rng.choice(values), 1)
            # Handle special reasoning-specific placeholders
            special = {
                "{claim}": self.rng.choice([
                    "The statement holds for all natural numbers n >= 1",
                    "The given expression equals the closed form for all valid inputs",
                    "The set has the stated property under the given conditions",
                ]),
                "{base_case}": self.rng.choice([
                    "For n = 1: LHS = 1, RHS = 1(1+1)/2 = 1. The base case holds.",
                    "For n = 0: The statement is trivially true since the sum is empty.",
                    "For n = 1: Direct computation verifies the identity.",
                ]),
                "{base_value}": self.rng.choice(["1", "0", "2"]),
                "{inductive_hypothesis}": self.rng.choice([
                    "the sum 1 + 2 + ... + k = k(k+1)/2",
                    "the statement holds for all values up to and including k",
                    "P(k) is true for our arbitrary but fixed k",
                ]),
                "{inductive_step}": self.rng.choice([
                    "Starting from the left side for k+1, we can factor out the inductive hypothesis and simplify algebraically.",
                    "We add the (k+1)-th term to both sides of the inductive hypothesis and show the resulting expression equals the claimed formula at k+1.",
                    "By the strong inductive hypothesis, we know the result for all values up to k. Applying this to the k+1 case gives us the desired result.",
                ]),
                "{quantifier}": self.rng.choice(["n >= 1", "positive integers n", "n >= 0", "integers n >= 2"]),
                "{key_insight}": self.rng.choice([
                    "recognizing the telescoping structure",
                    "the inductive step works because the recurrence adds exactly one term",
                    "factoring the expression reveals a clean closed form",
                    "the contradiction arises from the irrationality of the quantity",
                ]),
                "{proof_type}": self.rng.choice(["constructive", "non-constructive", "inductive", "algebraic"]),
                "{negation}": self.rng.choice([
                    "the quantity is rational, i.e., it can be written as p/q in lowest terms",
                    "there are only finitely many primes",
                    "the statement fails for some specific n = k",
                ]),
                "{consequence1}": "this implies a chain of algebraic identities",
                "{consequence2}": "after simplification, both sides must be divisible by the same prime",
                "{consequence3}": "this contradicts our assumption that the fraction was in lowest terms",
                "{original_assumption}": "the starting hypothesis",
                "{conclusion}": "the original statement must be true",
                "{contradiction_source}": "the assumption led to a statement that contradicts a known fact",
                "{significance}": "it demonstrates that proof by contradiction is a powerful technique when direct proof is difficult",
                "{variables}": "a, b be arbitrary elements satisfying the given conditions",
                "{proof_step1}": "By the given hypothesis, we know the initial conditions hold.",
                "{intermediate1}": "an expression relating the key quantities",
                "{intermediate2}": "a simplified form that directly implies the result",
                "{proof_step3}": "Combining the above, we obtain the desired inequality.",
                "{combination}": "the chain of equalities/inequalities gives us the final result",
                "{technique_name}": "direct construction combined with algebraic manipulation",
                "{technique_reason}": "the algebraic structure is well-suited to this approach",
                "{common_mistake}": "assume the result for n and try to prove it for n+1 without verifying the base case",
                "{mistake_reason}": "without the base case, the induction has no foundation",
                "{method}": "a combination of algebraic manipulation and logical reasoning",
                "{definition}": "a group is a set with a binary operation satisfying closure, associativity, identity, and inverse properties",
                "{lemma_statement}": "the auxiliary result needed for the main proof",
                "{lemma_proof}": "this follows directly from the definitions and basic properties",
                "{main_proof}": "using the lemma, the main result follows by applying the key technique",
                "{lemma_necessity}": "it isolates the technical core of the argument",
                "{gap_from}": "the hypothesis",
                "{gap_to}": "the conclusion",
                # Logic-specific
                "{given_info}": "the statements made by each person and the knight/knave constraint",
                "{case1_assumption}": "Alice is a knight (truth-teller)",
                "{case1_reasoning}": "Then her statement is true, which means...",
                "{case1_result}": "a consistent assignment (or a contradiction)",
                "{case2_assumption}": "Alice is a knave (liar)",
                "{case2_reasoning}": "Then her statement is false, which means...",
                "{case2_result}": "a consistent assignment (or a contradiction)",
                "{case_conclusion}": "Only one case is consistent, giving us the unique solution.",
                "{validity_reason}": "we exhaustively checked all logically possible cases",
                "{symbolization}": "Let P = the first condition, Q = the second condition",
                "{logical_form}": "P -> Q, P, therefore Q (modus ponens)",
                "{evaluation}": "Checking the truth table confirms validity",
                "{validity}": "valid",
                "{validity_explanation}": "there is no truth assignment making the premises true and the conclusion false",
                "{additional_note}": self.rng.choice(LOGICAL_PRINCIPLES),
                "{clue_ref1}": "1",
                "{clue_ref2}": "2",
                "{clue_ref3}": "3",
                "{deduction1}": "This eliminates two of the initial possibilities.",
                "{deduction2}": "The remaining options are further constrained.",
                "{deduction3}": "Only one assignment is now consistent.",
                "{deduction4}": "All other variables are determined.",
                "{verification}": "Checking each original clue against our solution confirms it works.",
                "{answer}": "the unique consistent assignment derived above",
                # Algorithm-specific
                "{pseudocode}": "function solve(input):\n    // Base case\n    if size(input) <= 1: return trivial_answer\n    // Recursive case\n    mid = size(input) / 2\n    left = solve(input[0..mid])\n    right = solve(input[mid..n])\n    return combine(left, right)",
                "{invariant}": "At the start of each iteration i, the subarray A[0..i-1] contains the correct sorted elements.",
                "{init_proof}": "Before the first iteration, the subarray is empty, so the invariant holds vacuously.",
                "{maintenance_proof}": "If the invariant holds at iteration i, then after placing A[i] in its correct position, the subarray A[0..i] is sorted.",
                "{termination_proof}": "The loop terminates when i = n. By the invariant, A[0..n-1] is the full sorted array.",
                "{time_analysis}": "Each level of recursion does O(n) work, and there are O(log n) levels, giving O(n log n) total.",
                "{space_analysis}": "O(n) auxiliary space for the merge buffer, or O(log n) for the recursion stack.",
                "{improvement_reason}": "it eliminates redundant comparisons by exploiting the divide-and-conquer structure",
                "{recurrence_statement}": "T(n) = 2T(n/2) + cn, with T(1) = 1",
                "{method_name}": "the Master Theorem",
                "{solve_step1}": "Identify a = 2, b = 2, f(n) = cn.",
                "{solve_step2}": "Compute log_b(a) = log_2(2) = 1. Compare with f(n) = O(n^1).",
                "{solve_step3}": "Since f(n) = Theta(n^(log_b(a))), we are in Case 2 of the Master Theorem.",
                "{closed_form}": "Theta(n log n)",
                "{complexity_class}": "O(n log n)",
                "{intuition}": "each element is compared O(log n) times across the recursion levels",
                "{source_problem}": "3-SAT",
                "{target_problem}": "the given problem",
                "{construction}": "For each variable x_i, create a gadget G_i. For each clause C_j, create a connector that enforces the clause constraint.",
                "{forward_proof}": "a satisfying assignment for I gives a valid solution for I'",
                "{backward_proof}": "a valid solution for I' can be decoded into a satisfying assignment for I",
                "{construction_time}": "O(n + m) where n is variables and m is clauses",
                "{time_justification}": "each variable and clause produces a constant-size gadget",
                "{approach_a_name}": "brute force",
                "{time_a}": "O(n^2)",
                "{space_a}": "O(1)",
                "{best_a}": "n is small or simplicity is paramount",
                "{approach_b_name}": "divide and conquer",
                "{time_b}": "O(n log n)",
                "{space_b}": "O(n)",
                "{best_b}": "n is large and we can afford the extra space",
                "{crossover}": "approaches break even around n ≈ 50-100 depending on constant factors",
                "{recommendation}": "Use Approach B for any production workload",
                "{practical_note}": "cache performance and branch prediction can cause up to 3x differences beyond what Big-O predicts",
                # Game theory
                "{best_response_a}": "If B plays High, A prefers Low (higher payoff). If B plays Low, A prefers Low.",
                "{best_response_b}": "By symmetry (or similar analysis), B also prefers Low regardless of A's choice.",
                "{mutual_best}": "Both playing Low is the only mutual best response.",
                "{equilibria}": "(Low, Low) is the unique Nash equilibrium.",
                "{pd_analysis}": "Yes — both players have a dominant strategy (Low), but mutual High would give both a higher payoff. This is the defining feature of a prisoner's dilemma.",
                "{efficiency_analysis}": "The Nash equilibrium is Pareto-dominated by (High, High), illustrating the tension between individual and collective rationality.",
                "{real_world}": "price competition, arms races, and public goods provision",
                "{game_tree}": "Player 1 moves first (L or R), then Player 2 observes and chooses (l or r).",
                "{final_node}": "Player 2 picks the action that maximizes their payoff.",
                "{penultimate_node}": "Anticipating Player 2's response, Player 1 evaluates their options.",
                "{first_node}": "Player 1 chooses the action that maximizes payoff given Player 2's best response.",
                "{spe}": "The subgame-perfect equilibrium strategy profile",
                "{path}": "Player 1 plays the dominant first move, Player 2 responds optimally.",
                "{payoffs}": "The resulting payoffs under the SPE strategy profile.",
                "{strategic_insight}": "the first mover can exploit their information advantage",
                "{observation}": "Player 2's threat to play aggressively is not credible in the SPE",
                "{principle}": "the concept of credible commitments in sequential games",
                "{mechanism_summary}": "a second-price sealed-bid auction (Vickrey auction)",
                "{ic_proof}": "If v' > v_i, the agent might win and pay more than their value (negative surplus). If v' < v_i, the agent might lose auctions they should have won. Either way, deviation doesn't help.",
                "{ir_proof}": "Each winning agent pays the second-highest bid, which is at most their own value. Losing agents pay nothing. So all agents have non-negative utility.",
                "{revenue_analysis}": "Revenue equals the second-highest valuation. By the revenue equivalence theorem, this yields the same expected revenue as a first-price auction.",
                "{related_mechanism}": "VCG mechanism",
                "{mechanism_principle}": "truthfulness can be a dominant strategy when payments are designed correctly",
                # Causal
                "{dag_description}": "X -> M -> Y, with C -> X and C -> Y (confounder C, mediator M)",
                "{paths}": "Direct: X -> M -> Y. Back-door: X <- C -> Y.",
                "{confounders}": "C is a confounder because it causes both X and Y through separate paths.",
                "{adjustment_set}": "{C}",
                "{adjustment_justification}": "Controlling for C blocks the back-door path X <- C -> Y while leaving the causal path X -> M -> Y open.",
                "{no_control}": "M (the mediator)",
                "{no_control_reason}": "controlling for a mediator blocks the causal effect we're trying to estimate",
                "{identification}": "The causal effect is identifiable by the back-door criterion after adjusting for C.",
                "{assumptions}": "no unmeasured confounders, correct DAG specification, and positivity",
                "{weak_assumption}": "no unmeasured confounders",
                "{weak_reason}": "we can never fully rule out variables we haven't measured",
                "{observed_diff}": "Group A outperformed Group B by the observed margin.",
                "{randomization_check}": "Was assignment truly random? Check for baseline balance across groups.",
                "{sample_check}": "With N subjects per group, we have sufficient power to detect medium effects.",
                "{attrition_check}": "Check whether dropout rates differ between groups.",
                "{compliance_check}": "Did all subjects actually receive their assigned treatment?",
                "{threats}": "Hawthorne effect, novelty effect, and potential SUTVA violations (spillover).",
                "{statistical_analysis}": "A two-sample test gives us the p-value and confidence interval for the treatment effect.",
                "{strength}": "random assignment eliminates confounding in expectation",
                "{caveat}": "we must still consider external validity (generalizability) and effect heterogeneity",
                "{causal_fallacy}": "confusing correlation with causation",
                "{original_claim}": "the stated causal relationship",
                "{flaw1_name}": "Omitted variable bias",
                "{flaw1_detail}": "There may be a common cause driving both variables.",
                "{flaw2_name}": "Reverse causation",
                "{flaw2_detail}": "The direction of causality may be opposite to what was claimed.",
                "{flaw3_name}": "Selection bias",
                "{flaw3_detail}": "The sample may not be representative of the population.",
                "{fix_description}": "A randomized experiment or careful instrumental variable analysis could address these issues.",
                "{correct_statement}": "Based on the available evidence, we can only say there is an association, not a causal relationship.",
                # Science
                "{hypothesis}": "the proposed causal mechanism",
                "{iv}": "the factor being manipulated",
                "{dv}": "the outcome being measured",
                "{controls}": "all other factors held constant across groups",
                "{design_type}": "a randomized controlled trial with double-blinding",
                "{proc_step1}": "Randomly assign participants to treatment and control groups.",
                "{proc_step2}": "Administer the intervention to the treatment group and a placebo to the control.",
                "{proc_step3}": "Measure outcomes at pre-specified time points and compare groups.",
                "{confounders}": "age, baseline severity, and socioeconomic status as potential confounders",
                "{stat_plan}": "intention-to-treat analysis with pre-registered primary endpoints",
                "{power_analysis}": "With n=200 per group, we have 80% power to detect an effect size of d=0.3.",
                "{strength}": "randomization eliminates confounding in expectation",
                "{limitation}": "the artificial lab setting may not generalize to real-world conditions",
                "{mitigation}": "including a naturalistic follow-up phase",
                "{correlation}": "the observed statistical association",
                "{causal_argument}": "there is a plausible biological mechanism and the effect is dose-dependent",
                "{reverse}": "the outcome might influence the exposure rather than vice versa",
                "{confounding}": "a third variable (e.g., SES, genetics) might cause both",
                "{selection_bias}": "people in the exposed group may differ systematically from the unexposed",
                "{artifact}": "the measurement tool might be biased or the analysis might be underpowered",
                "{causation_criteria}": "temporal precedence, mechanism, dose-response, consistency across studies, and ideally an RCT",
                "{fundamental_issue}": "observational data alone cannot establish causation without additional assumptions",
                "{methodological_principle}": "we use randomized experiments as the gold standard for causal inference",
                "{evidence_for}": "multiple studies showing consistent results, dose-response relationship, plausible mechanism",
                "{evidence_against}": "conflicting results in some populations, potential confounders not fully addressed",
                "{quality_assessment}": "the strongest evidence comes from the RCTs; observational studies are more numerous but weaker",
                "{assessment}": "the weight of evidence moderately supports the claim, with important caveats",
                "{critical_factor}": "the quality and consistency of the underlying studies",
                "{framework}": "a Bayesian evidence synthesis framework",
                "{confidence_level}": "70-75%",
                "{confidence_reason}": "while the evidence is suggestive, key alternative explanations haven't been fully ruled out",
                # Philosophy
                "{core_tension}": "the conflict between competing intuitions about the nature of the concept",
                "{position_a_name}": "the standard view",
                "{position_a_argument}": "The traditional argument rests on the premises that...",
                "{position_b_name}": "the revisionist view",
                "{position_b_argument}": "The alternative argument challenges the traditional view by...",
                "{a_strength}": "it captures our pre-theoretical intuitions well",
                "{a_weakness}": "it faces clear counterexamples in edge cases",
                "{b_weakness}": "it requires giving up some deeply-held intuitions",
                "{preferred_position}": "a qualified version of Position A",
                "{preference_reason}": "it handles the most important cases correctly while being revisable in edge cases",
                "{critical_assumption}": "our intuitions about thought experiments are reliable guides to truth",
                "{premise1}": "the first premise of the reconstructed argument",
                "{premise2}": "the second premise",
                "{premise3}": "the third premise (often a hidden assumption)",
                "{eval1}": "This premise is widely accepted and well-supported.",
                "{eval2}": "This premise is more controversial; it relies on the assumption that...",
                "{eval3}": "This is the weakest premise — it is essentially the point at issue.",
                "{validity_detail}": "follows from the premises by valid logical rules",
                "{soundness}": "questionable",
                "{soundness_reason}": "Premise 3 begs the question",
                "{historical_note}": "This argument has a long history, dating back to the ancient Greeks. Modern formulations address some but not all classical objections.",
                "{utilitarian_analysis}": "The total expected utility calculation yields...",
                "{utilitarian_verdict}": "permissible (maximizes aggregate welfare)",
                "{deontological_analysis}": "Under the categorical imperative, we must ask whether the maxim can be universalized...",
                "{deontological_verdict}": "impermissible (uses persons merely as means)",
                "{virtue_analysis}": "The virtuous agent would exhibit compassion, justice, and practical wisdom...",
                "{virtue_verdict}": "depends on the character and motivations of the agent",
                "{synthesis}": "the frameworks disagree in this case, which is what makes it a genuine ethical dilemma",
                "{agreement_status}": "reach different conclusions",
                "{meta_insight}": "no single ethical framework captures all of our moral intuitions",
                "{fundamental_disagreement}": "whether consequences or principles should take priority",
                "{deep_question}": "what we owe to each other as moral agents",
                # Puzzle
                "{analysis}": "First, let me determine the information content of the problem.",
                "{strategy}": "The optimal strategy uses a divide-and-conquer approach on the possibilities.",
                "{step1}": "Divide the items into groups and perform the first comparison.",
                "{step2}": "Based on the result, narrow down to the relevant subset.",
                "{step3}": "One more comparison identifies the exact item.",
                "{optimality}": "With N items and 3 possible outcomes per weighing (left heavy, balanced, right heavy), we need at least ceiling(log_3(N)) weighings. Our solution achieves this bound.",
                "{constraints}": "The puzzle constraints form a system of logical equations.",
                "{c1}": "A", "{c2}": "B", "{c3}": "C",
                "{solution}": "The unique solution satisfying all constraints.",
                "{uniqueness_proof}": "We've shown that every constraint narrows to exactly one possibility, so the solution is unique.",
                "{full_verification}": "Checking constraint by constraint: all satisfied.",
                "{total}": "the full combinatorial space of possibilities",
                "{elimination1}": "half the space is immediately ruled out",
                "{remaining1}": "a much smaller set",
                "{elimination2}": "further narrowing",
                "{remaining2}": "a handful of candidates",
                "{elimination3}": "the final constraint pins down the answer",
                "{remaining3}": "exactly 1",
                "{result}": "the unique solution",
                "{difficulty_source}": "the combinatorial explosion of possibilities makes brute force impractical",
                "{crucial_step}": "finding the right order in which to apply constraints for maximum pruning",
            }
            for key, value in special.items():
                if isinstance(value, str):
                    if key in result:
                        result = result.replace(key, value, 1)
                elif isinstance(value, list):
                    if key in result:
                        result = result.replace(key, self.rng.choice(value), 1)
            # Handle reasoning-specific content blocks
            if "{proof_technique}" in result:
                result = result.replace("{proof_technique}", self.rng.choice(PROOF_TECHNIQUES), 1)
            if "{reasoning_chain}" in result:
                result = result.replace("{reasoning_chain}", self.rng.choice(REASONING_CHAINS), 1)
            if "{math_concept}" in result:
                result = result.replace("{math_concept}", self.rng.choice(MATHEMATICAL_CONCEPTS), 1)
            if "{logical_principle}" in result:
                result = result.replace("{logical_principle}", self.rng.choice(LOGICAL_PRINCIPLES), 1)
            if "{complexity_result}" in result:
                result = result.replace("{complexity_result}", self.rng.choice(COMPLEXITY_RESULTS), 1)
            # Break if no more placeholders to fill
            if "{" not in result:
                break
        # Final cleanup: remove any remaining unfilled placeholders
        result = re.sub(r'\{[a-zA-Z_0-9]+\}', '', result)
        return result.strip()

    def _pick_topic(self) -> dict:
        """Weighted random topic selection."""
        return self.rng.choices(self.topics, weights=self.topic_weights, k=1)[0]

    def _response_length_bucket(self, turn_index: int) -> str:
        """Determine response length based on turn position."""
        dist_config = self.config["response_length"]["length_distribution_by_turn"]
        if turn_index < 5:
            dist = dist_config["early"]
        elif turn_index < 20:
            dist = dist_config["middle"]
        else:
            dist = dist_config["late"]
        buckets = list(dist.keys())
        weights = list(dist.values())
        return self.rng.choices(buckets, weights=weights, k=1)[0]

    def _generate_response_text(self, topic_name: str, turn_index: int) -> str:
        """Generate an assistant response of appropriate length."""
        templates = TOPIC_TEMPLATES[topic_name]["responses"]
        base = self._fill_template(self.rng.choice(templates), topic_name)

        # Determine target length
        bucket = self._response_length_bucket(turn_index)
        length_config = self.config["response_length"][bucket]
        target_words = self.rng.randint(length_config["min_words"], length_config["max_words"])

        # Pad or trim to approximate target
        words = base.split()
        if len(words) < target_words:
            # Extend with reasoning-appropriate filler
            extensions = [
                "\n\nLet me elaborate on why this step is necessary. Without it, we'd be making an unjustified logical leap that could invalidate the entire argument.",
                "\n\nTo build more intuition, consider what happens in the boundary case. The behavior there often illuminates the general principle.",
                "\n\nIt's worth pausing to consider whether our approach generalizes. The technique we used here relies on a specific structural property that may or may not hold in broader contexts.",
                "\n\nA common error at this point is to assume the converse is also true. But the converse requires a separate proof — implication is not bidirectional by default.",
                "\n\nNote that this result is tight — meaning we cannot improve the bound without additional assumptions. I can show this by constructing a matching lower bound example.",
                "\n\nThe elegance of this approach lies in its reduction: we transformed a complex problem into one we already know how to solve. This is one of the most powerful techniques in mathematical reasoning.",
                "\n\nTo verify our answer, let's check it against the original constraints. Substituting back, we get consistent values at every step, confirming correctness.",
                "\n\nThis connects to a deeper principle: when the structure of a problem has symmetry, we can often exploit that symmetry to simplify the analysis dramatically.",
                "\n\nConsider the contrapositive direction as well. If the conclusion were false, then by tracing back through our chain of implications, we would necessarily violate one of the premises.",
                "\n\nFor completeness, let me address the edge cases. When n = 0 or n = 1, the result holds trivially. For n >= 2, the general argument applies.",
            ]
            while len(words) < target_words:
                ext = self.rng.choice(extensions)
                words.extend(ext.split())
        elif len(words) > target_words * 1.3:
            words = words[:target_words]
            # Try to end on a sentence boundary
            text = " ".join(words)
            last_period = text.rfind(".")
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]
            return text

        return " ".join(words)

    def _generate_user_message(self, topic_name: str, turn_index: int) -> str:
        """Generate a user message: opener for first turn, followup otherwise."""
        templates = TOPIC_TEMPLATES[topic_name]
        if turn_index == 0:
            template = self.rng.choice(templates["openers"])
        else:
            template = self.rng.choice(templates["followups"])
        return self._fill_template(template, topic_name)

    def generate_conversation(self, num_turns: int) -> dict:
        """Generate a single multi-turn deep reasoning conversation."""
        topic = self._pick_topic()
        topic_name = topic["name"]
        system_prompt = topic["system_prompt"]

        messages = [{"role": "system", "content": system_prompt}]
        cumulative_char_lengths = []
        running_chars = len(system_prompt)

        for turn_idx in range(num_turns):
            # User message
            user_msg = self._generate_user_message(topic_name, turn_idx)
            messages.append({"role": "user", "content": user_msg})
            running_chars += len(user_msg)

            # Assistant response
            assistant_msg = self._generate_response_text(topic_name, turn_idx)
            messages.append({"role": "assistant", "content": assistant_msg})
            running_chars += len(assistant_msg)

            cumulative_char_lengths.append(running_chars)

        conversation_id = str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

        return {
            "conversation_id": conversation_id,
            "topic": topic_name,
            "num_turns": num_turns,
            "num_messages": len(messages),
            "system_prompt": system_prompt,
            "messages": json.dumps(messages),
            "total_characters": running_chars,
            "estimated_tokens": running_chars // 4,  # rough approximation
            "cumulative_char_lengths": json.dumps(cumulative_char_lengths),
        }

    def generate_dataset(self, num_conversations: int = None) -> list[dict]:
        """Generate the full dataset according to config distribution."""
        if num_conversations is not None:
            # Override: uniform random turns
            conversations = []
            turn_min = self.config["turns"]["min"]
            turn_max = self.config["turns"]["max"]
            for _ in range(num_conversations):
                n_turns = self.rng.randint(turn_min, turn_max)
                conversations.append(self.generate_conversation(n_turns))
            return conversations

        # Use configured distribution buckets
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
# Export formats
# ---------------------------------------------------------------------------

def convert_to_aiperf_multi_turn(conversations: list[dict]) -> list[dict]:
    """Convert conversations to aiperf multi_turn JSONL format.

    aiperf multi_turn format (one JSON object per line):
        {"session_id": "...", "turns": [{"text": "user msg"}, {"text": "user msg 2"}, ...]}

    This uses the `deltas_without_responses` context mode — aiperf sends only
    user messages and accumulates live server responses into conversation history
    automatically. This is ideal for benchmarking prefix caching because each
    turn's request includes the full conversation prefix.

    Usage with aiperf:
        aiperf profile \\
            --model <model> \\
            --endpoint-type chat \\
            --input-file multi_turn_reasoning_chat.jsonl \\
            --custom-dataset-type multi_turn \\
            --streaming --url localhost:8000

    Returns:
        List of dicts, each representing one conversation line for JSONL output.
    """
    aiperf_entries = []
    for conv in conversations:
        messages = json.loads(conv["messages"])
        turns = []
        for msg in messages:
            if msg["role"] == "user":
                turns.append({"text": msg["content"]})
        if turns:
            aiperf_entries.append({
                "session_id": conv["conversation_id"],
                "turns": turns,
            })
    return aiperf_entries


def convert_to_aiperf_mooncake(conversations: list[dict], block_size: int = 512) -> list[dict]:
    """Convert conversations to aiperf mooncake_trace JSONL format.

    This format sends the full message array (system + user + assistant) per turn,
    using the `message_array_with_responses` context mode. Each line represents a
    single turn within a session, with the complete conversation history up to that
    point. This gives full control over the exact prompt sent to the server.

    Format (one JSON object per line):
        {"session_id": "...", "messages": [...], "output_length": N}

    Usage with aiperf:
        aiperf profile \\
            --model <model> \\
            --endpoint-type chat \\
            --input-file multi_turn_reasoning_chat_mooncake.jsonl \\
            --custom-dataset-type mooncake_trace \\
            --streaming --url localhost:8000

    Returns:
        List of dicts, one per turn across all conversations.
    """
    entries = []
    for conv in conversations:
        messages = json.loads(conv["messages"])
        session_id = conv["conversation_id"]
        # Walk through the conversation, building up context each turn
        context = []
        for i, msg in enumerate(messages):
            context.append(msg)
            # Emit an entry after each assistant message (= end of a turn pair)
            if msg["role"] == "assistant":
                # Estimate output length for this assistant response
                output_tokens = max(1, len(msg["content"]) // 4)
                entry = {
                    "session_id": session_id,
                    "messages": [m for m in context],  # full context up to here
                    "output_length": output_tokens,
                }
                # Add delay for turns after the first (simulate user think time)
                turn_index = len([m for m in context if m["role"] == "assistant"])
                if turn_index > 1:
                    entry["delay"] = 0  # no artificial delay; set >0 to simulate think time
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-turn deep reasoning chat dataset"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--num", type=int, default=None, help="Override number of conversations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--format", choices=["all", "parquet", "aiperf", "mooncake"],
                        default="all", help="Output format(s): parquet, aiperf (multi_turn JSONL), "
                        "mooncake (mooncake_trace JSONL), or all (default)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config["dataset"]["seed"]
    generator = ReasoningConversationGenerator(config, seed=seed)

    print(f"Generating deep reasoning conversations (seed={seed})...")
    conversations = generator.generate_dataset(num_conversations=args.num)
    print(f"Generated {len(conversations)} conversations")

    # Build DataFrame
    df = pd.DataFrame(conversations)

    # Summary statistics
    print(f"\n--- Dataset Summary ---")
    print(f"Total conversations: {len(df)}")
    print(f"Turn count range: {df['num_turns'].min()} - {df['num_turns'].max()}")
    print(f"Mean turns: {df['num_turns'].mean():.1f}")
    print(f"Topic distribution:")
    for topic, count in df["topic"].value_counts().items():
        print(f"  {topic}: {count} ({100*count/len(df):.1f}%)")
    print(f"Estimated total tokens: {df['estimated_tokens'].sum():,}")
    print(f"Mean tokens/conversation: {df['estimated_tokens'].mean():,.0f}")
    print(f"Max tokens (single conversation): {df['estimated_tokens'].max():,}")

    # Determine output directory
    output_dir = Path(args.output).parent if args.output else Path(__file__).parent / config["dataset"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format

    # Write Parquet
    if fmt in ("all", "parquet"):
        parquet_path = Path(args.output) if args.output and fmt == "parquet" else output_dir / config["dataset"]["output_filename"]
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"\nParquet written to: {parquet_path} ({file_size_mb:.2f} MB)")

    # Write aiperf multi_turn JSONL
    if fmt in ("all", "aiperf"):
        aiperf_entries = convert_to_aiperf_multi_turn(conversations)
        jsonl_path = output_dir / config["dataset"]["output_filename"].replace(".parquet", ".jsonl")
        with open(jsonl_path, "w") as f:
            for entry in aiperf_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_size_mb = jsonl_path.stat().st_size / (1024 * 1024)
        print(f"aiperf multi_turn JSONL written to: {jsonl_path} ({file_size_mb:.2f} MB)")
        print(f"  Usage: aiperf profile --input-file {jsonl_path} --custom-dataset-type multi_turn ...")

    # Write aiperf mooncake_trace JSONL
    if fmt in ("all", "mooncake"):
        mooncake_entries = convert_to_aiperf_mooncake(conversations)
        mooncake_path = output_dir / config["dataset"]["output_filename"].replace(".parquet", "_mooncake.jsonl")
        with open(mooncake_path, "w") as f:
            for entry in mooncake_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_size_mb = mooncake_path.stat().st_size / (1024 * 1024)
        print(f"aiperf mooncake_trace JSONL written to: {mooncake_path} ({file_size_mb:.2f} MB)")
        print(f"  Usage: aiperf profile --input-file {mooncake_path} --custom-dataset-type mooncake_trace ...")


if __name__ == "__main__":
    main()
