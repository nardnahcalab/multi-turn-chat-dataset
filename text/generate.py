#!/usr/bin/env python3
"""
Synthetic multi-turn text conversation generator for inference benchmarking.

Generates realistic conversations across multiple domains with naturally growing
context, designed to stress-test prefix caching in LLM inference engines.

Usage:
    python generate.py                     # uses default config.yaml
    python generate.py --config my.yaml    # custom config
    python generate.py --num 1000          # override conversation count
"""

import argparse
import hashlib
import json
import random
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
    "customer_support": {
        "openers": [
            "I ordered a {product} {timeframe} and it still hasn't arrived. Order number is {order_id}. Can you help?",
            "I was charged twice for my recent order #{order_id}. I need a refund for the duplicate charge.",
            "The {product} I received is defective — {defect}. I'd like to return or exchange it.",
            "I'm trying to cancel my subscription but can't find the option in my account settings.",
            "I placed an order for a {product} but I need to change the shipping address before it ships.",
            "My account seems to be locked after too many failed login attempts. Username is {username}.",
            "I received the wrong item in my order #{order_id}. I ordered a {product} but got a {wrong_product}.",
            "I have a question about the warranty on my {product}. Does it cover {defect}?",
            "I'd like to apply a promo code to an order I already placed. Is that possible? Order #{order_id}.",
            "The tracking information for my order #{order_id} hasn't updated in {timeframe}. Is it lost?",
        ],
        "followups": [
            "Yes, my email on file is {email}. Can you look into it?",
            "I've already tried that. It didn't resolve the issue.",
            "That's not what I was told by the previous agent. Can you escalate this?",
            "How long will the {action} take to process?",
            "Can I get a confirmation email once that's done?",
            "Actually, I have another issue with the same order — {issue}.",
            "Is there a way to expedite this? I need the {product} by {date}.",
            "What are my options if the {action} doesn't work?",
            "I'd also like to update my {detail} while we're at it.",
            "Thank you, that helps. One more thing — {question}",
            "I checked and the {detail} still shows the old information.",
            "Can you send me a return shipping label?",
            "Do I need to return the defective item before getting the replacement?",
            "What's the best way to reach you if this isn't resolved?",
        ],
        "responses": [
            "I understand your frustration with the {issue}. Let me look into this right away.\n\nI've pulled up your order #{order_id} and I can see that {finding}. Here's what I'll do to resolve this:\n\n1. {step1}\n2. {step2}\n3. {step3}\n\nThis should be completed within {timeline}. Is there anything else I can help with?",
            "Thank you for your patience. I've reviewed your account and here's what I found:\n\n{finding}\n\nTo resolve this, I've {action_taken}. You should see the changes reflected in {timeline}.",
            "I'm sorry about the inconvenience. Let me walk you through the next steps:\n\n- {step1}\n- {step2}\n\nIf you run into any issues, you can reference case number {case_id}.",
            "Great news — I was able to {action_taken}. Here's a summary:\n\n**Order:** #{order_id}\n**Action:** {step1}\n**Expected Timeline:** {timeline}\n**Confirmation:** A notification has been sent to {email}\n\nPlease let me know if you need anything else.",
            "I've escalated this to our {department} team. They'll review it within {timeline}. In the meantime, here's what you should know:\n\n{finding}\n\nYour reference number is {case_id}.",
        ],
        "fill_values": {
            "product": ["laptop stand", "wireless headphones", "ergonomic keyboard", "monitor arm",
                       "USB-C hub", "webcam", "standing desk", "mesh office chair", "tablet case",
                       "portable charger", "Bluetooth speaker", "smart watch", "noise-canceling earbuds"],
            "wrong_product": ["phone case", "mouse pad", "cable organizer", "screen protector"],
            "timeframe": ["3 days ago", "a week ago", "over 10 days ago", "two weeks ago"],
            "defect": ["the screen is cracked", "it won't turn on", "a button is stuck",
                      "the color is wrong", "it's missing parts", "the hinge is broken"],
            "action": ["refund", "replacement", "exchange", "address change", "cancellation"],
            "detail": ["shipping address", "payment method", "email address", "phone number"],
            "department": ["logistics", "billing", "technical support", "quality assurance"],
            "timeline": ["24-48 hours", "1-2 business days", "3-5 business days", "within the hour"],
        },
    },

    "coding_help": {
        "openers": [
            "I'm getting a `{error}` when I try to {action} in my {language} project. Here's the relevant code:\n\n```{language}\n{code_snippet}\n```\n\nWhat am I doing wrong?",
            "Can you help me design a {pattern} for my {project_type}? I need to handle {requirement}.",
            "I'm trying to optimize this {language} function that's running too slowly on large inputs:\n\n```{language}\n{code_snippet}\n```\n\nIt takes {duration} for {input_size} records. How can I speed it up?",
            "What's the best way to implement {feature} in {framework}? I've looked at {approach} but I'm not sure it's the right pattern.",
            "I need to write a {language} function that {task}. What's the cleanest approach?",
            "I'm reviewing this code and I think there might be a race condition:\n\n```{language}\n{code_snippet}\n```\n\nCan you verify and suggest a fix?",
            "How should I structure the database schema for {project_type}? The main entities are {entities}.",
            "I need to set up CI/CD for my {project_type}. We're using {tools}. What's a good pipeline config?",
        ],
        "followups": [
            "That makes sense. But what about the edge case where {edge_case}?",
            "I tried your suggestion and now I'm getting a different error: `{error}`",
            "Can you show me how to write a unit test for this?",
            "What about thread safety? Multiple workers will call this concurrently.",
            "How would this change if I need to support {requirement}?",
            "Can you refactor this to use {pattern} instead?",
            "What's the time complexity of your solution?",
            "I also need to handle the case where {edge_case}. Should I add that to the same function or create a separate one?",
            "How would I add logging and error handling to this?",
            "The tests pass locally but fail in CI. The error is `{error}`. Ideas?",
            "Can you explain why {approach} is better than {alt_approach} here?",
            "I need to add authentication to this endpoint. What's the recommended approach with {framework}?",
            "How should I handle database migrations for this schema change?",
            "What about caching? This endpoint gets called {frequency} and the data changes {change_freq}.",
        ],
        "responses": [
            "The issue is with {diagnosis}. Here's what's happening:\n\n{explanation}\n\nHere's the fix:\n\n```{language}\n{fixed_code}\n```\n\nThe key change is {key_change}. This works because {reasoning}.",
            "I'd recommend the following approach:\n\n```{language}\n{fixed_code}\n```\n\n**Why this works:**\n- {reason1}\n- {reason2}\n- {reason3}\n\nFor the edge case you mentioned, you'd want to add a check like:\n\n```{language}\n{edge_case_code}\n```",
            "Good question. There are a few approaches:\n\n**Option A: {approach_a}**\nPros: {pros_a}\nCons: {cons_a}\n\n**Option B: {approach_b}**\nPros: {pros_b}\nCons: {cons_b}\n\nGiven your requirements ({requirement}), I'd go with Option {recommendation} because {reasoning}.",
            "Here's how I'd structure it:\n\n```{language}\n{fixed_code}\n```\n\nKey design decisions:\n1. {decision1}\n2. {decision2}\n3. {decision3}\n\nThis gives you {benefit} while keeping {constraint}.",
            "You're right — there is a race condition. The problem is {diagnosis}.\n\nHere's the thread-safe version:\n\n```{language}\n{fixed_code}\n```\n\nI used {pattern} here to ensure {guarantee}. Note that this adds a small overhead of {tradeoff}, but it's necessary for correctness.",
        ],
        "fill_values": {
            "language": ["Python", "TypeScript", "Go", "Rust", "Java", "C++"],
            "framework": ["FastAPI", "Express.js", "Spring Boot", "React", "Next.js", "Django"],
            "error": ["TypeError: cannot read property of undefined", "ConnectionRefusedError",
                      "IndexError: list index out of range", "deadlock detected",
                      "CORS policy blocked", "OOM killed", "segmentation fault",
                      "cannot borrow as mutable", "lifetime mismatch"],
            "pattern": ["repository pattern", "event-driven architecture", "CQRS pattern",
                       "strategy pattern", "observer pattern", "middleware chain"],
            "project_type": ["REST API", "microservice", "CLI tool", "web app",
                           "data pipeline", "real-time chat system"],
            "duration": ["30 seconds", "2 minutes", "over 5 minutes"],
            "input_size": ["100K", "1M", "10M"],
            "tools": ["GitHub Actions", "GitLab CI", "Jenkins", "ArgoCD + Kubernetes"],
            "frequency": ["~1000 times/sec", "~50 times/min", "~10K times/hour"],
            "change_freq": ["rarely", "every few minutes", "in real-time"],
        },
    },

    "creative_writing": {
        "openers": [
            "I'm working on a {genre} story set in {setting}. The main character is {character}. Can you help me develop the opening scene?",
            "I've written the first chapter of my novel but the pacing feels off. Here's the opening paragraph:\n\n\"{excerpt}\"\n\nHow can I improve it?",
            "I need help brainstorming a plot twist for my {genre} story. So far, {plot_summary}.",
            "Can you help me write a {form} about {theme}? I want it to feel {mood}.",
            "I'm struggling with dialogue for a scene where {scene_description}. The characters are {characters}.",
            "How do I show {emotion} without telling? My character just {event} and I want the reader to feel it.",
        ],
        "followups": [
            "I love that direction. Can you expand on the {element} part?",
            "What if we changed the setting to {setting}? How would that affect the tone?",
            "Can you rewrite that passage with more sensory details?",
            "The dialogue feels a bit stiff. Can you make it sound more natural?",
            "How should I transition from this scene to the next one where {next_scene}?",
            "I want to add a subplot about {subplot}. How do I weave it in?",
            "Can you write the next few paragraphs? I want to see where you'd take it.",
            "How can I make the antagonist more sympathetic without losing the tension?",
            "What's a good way to foreshadow {event} without being too obvious?",
            "I need a strong closing line for this chapter. Something that makes the reader want to keep going.",
        ],
        "responses": [
            "Here's a revised version of your opening that tightens the pacing and creates more immediate tension:\n\n\"{revised_excerpt}\"\n\nNotice how I {technique1}. This works because {reasoning}. For the next paragraph, you could {suggestion}.",
            "Let me draft a scene that captures what you're going for:\n\n---\n\n{scene_text}\n\n---\n\nA few things I did here:\n- {technique1}\n- {technique2}\n- {technique3}\n\nFeel free to adjust the tone — this is just a starting point.",
            "Great instinct to focus on {element}. Here are three approaches:\n\n**1. {approach1}**\n{example1}\n\n**2. {approach2}**\n{example2}\n\n**3. {approach3}**\n{example3}\n\nI'd lean toward #{recommendation} for your story because {reasoning}.",
            "For the {form} about {theme}, here's a draft:\n\n{creative_piece}\n\nThe structure mirrors {technique1}, which reinforces the {mood} tone. The last line echoes the opening to create a sense of {effect}.",
        ],
        "fill_values": {
            "genre": ["sci-fi", "literary fiction", "fantasy", "thriller", "historical fiction", "horror"],
            "setting": ["a dying space station", "1920s Paris", "a flooded city in 2140",
                       "a small town hiding a secret", "an underground kingdom", "a library that exists between worlds"],
            "form": ["poem", "short story", "flash fiction piece", "monologue", "prose poem"],
            "theme": ["loss and memory", "the passage of time", "finding home", "identity",
                     "the cost of ambition", "connection in isolation"],
            "mood": ["melancholic", "hopeful", "unsettling", "dreamlike", "raw and urgent"],
            "emotion": ["grief", "betrayal", "quiet joy", "mounting dread", "relief", "longing"],
        },
    },

    "tutoring": {
        "openers": [
            "I'm studying {subject} and I don't understand {concept}. Can you explain it in simple terms?",
            "I have an exam on {subject} next week and I'm struggling with {concept}. Can you walk me through it?",
            "Can you help me solve this {subject} problem?\n\n{problem_statement}",
            "What's the difference between {concept_a} and {concept_b}? I keep mixing them up.",
            "I'm learning {subject} and I just read about {concept}. Can you give me a real-world example of how it's used?",
            "I need to write a paper on {topic}. Can you help me build an outline and thesis statement?",
        ],
        "followups": [
            "That makes more sense now. But why does {follow_question}?",
            "Can you give me another example? I want to make sure I really get it.",
            "How does this connect to {related_concept}?",
            "What would happen if {hypothetical}?",
            "Can you show me the step-by-step solution? I got lost at the {step} step.",
            "How would I solve a similar problem but with {variation}?",
            "Is there a formula or mnemonic I can use to remember this?",
            "My teacher said {claim}. Is that correct?",
            "Can you quiz me on this to see if I understand?",
            "How is this topic likely to show up on the exam?",
            "I tried solving it and got {answer}. Where did I go wrong?",
            "What's the intuition behind {concept}? I can follow the steps but I don't understand why it works.",
        ],
        "responses": [
            "Great question! Let me break {concept} down:\n\n**In simple terms:** {simple_explanation}\n\n**Analogy:** Think of it like {analogy}\n\n**More precisely:** {precise_explanation}\n\nHere's a concrete example:\n{example}\n\nDoes that click? The key insight is {key_insight}.",
            "Let's work through this step by step:\n\n**Step 1:** {step1}\n**Step 2:** {step2}\n**Step 3:** {step3}\n**Step 4:** {step4}\n\nSo the answer is **{answer}**.\n\nThe trick here is recognizing that {key_insight}. Once you see that pattern, similar problems become straightforward.",
            "Good comparison — they're easy to confuse. Here's the key difference:\n\n| | {concept_a} | {concept_b} |\n|---|---|---|\n| Definition | {def_a} | {def_b} |\n| When to use | {use_a} | {use_b} |\n| Example | {ex_a} | {ex_b} |\n\nThe quick way to remember: {mnemonic}.",
            "Here's a real-world example:\n\n{real_world_example}\n\nThis demonstrates {concept} because {reasoning}. You see this pattern in {applications}.\n\nTo test your understanding: {mini_quiz}",
        ],
        "fill_values": {
            "subject": ["calculus", "linear algebra", "organic chemistry", "physics",
                       "machine learning", "microeconomics", "world history", "statistics"],
            "concept": ["eigenvalues", "the chain rule", "supply and demand equilibrium",
                       "photosynthesis", "backpropagation", "the French Revolution",
                       "Bayesian inference", "chemical bonding", "integration by parts"],
            "related_concept": ["matrix decomposition", "the product rule", "market failures",
                               "cellular respiration", "gradient descent", "the Industrial Revolution"],
            "concept_a": ["eigenvalues", "derivatives", "supply", "kinetic energy",
                         "precision", "classical conditioning"],
            "concept_b": ["eigenvectors", "integrals", "demand", "potential energy",
                         "recall", "operant conditioning"],
            "topic": ["the impact of the Industrial Revolution on urbanization",
                     "renewable energy policy in developing nations",
                     "the evolution of programming paradigms"],
        },
    },

    "travel_planning": {
        "openers": [
            "I'm planning a {duration} trip to {destination} in {month}. Budget is around ${budget}. Can you help me plan?",
            "My partner and I want to go somewhere in {region} for our anniversary. We like {interests}. Any suggestions?",
            "I have {duration} off work and want to do a solo backpacking trip. I've been to {visited}. Where should I go next?",
            "We're taking the kids ({ages}) to {destination}. What are the must-do family activities?",
            "I need to plan a business trip to {destination} next month. Can you recommend hotels near {area} and good restaurants for client dinners?",
        ],
        "followups": [
            "That sounds great! How should I split my time between {place_a} and {place_b}?",
            "What's the best way to get from the airport to {area}? Taxi, metro, or rideshare?",
            "Any restaurant recommendations for {cuisine} food in that area?",
            "What should I pack for the weather in {month}?",
            "Are there any day trips worth doing from {destination}?",
            "What about safety? Anything I should be aware of?",
            "How much should I budget daily for food and activities?",
            "Can you suggest a day-by-day itinerary?",
            "I also want to add {duration} in {nearby_place}. Is that doable?",
            "What's the best way to avoid tourist traps?",
            "Do I need to book anything in advance, or can I wing it?",
            "Is it worth getting a {pass_type} pass?",
        ],
        "responses": [
            "Great choice! {destination} in {month} is {weather_description}. Here's what I'd suggest:\n\n**Getting There:** {transport}\n**Where to Stay:** {accommodation}\n**Must-See:** {highlights}\n\nBudget breakdown:\n- Flights: ~${flight_cost}\n- Accommodation: ~${hotel_cost}/night\n- Food: ~${food_cost}/day\n- Activities: ~${activity_cost}/day\n\nTotal estimated: ~${total}",
            "Here's a day-by-day itinerary:\n\n**Day 1:** {day1}\n**Day 2:** {day2}\n**Day 3:** {day3}\n**Day 4:** {day4}\n\nPro tips:\n- {tip1}\n- {tip2}\n- {tip3}\n\nBook {advance_booking} in advance — they sell out fast.",
            "Based on your interests in {interests}, I'd recommend {destination}. Here's why:\n\n{reasoning}\n\nThe best time to visit is {best_time}, and you'll want at least {min_duration} to cover the highlights. Compared to {alternative}, it's {comparison}.",
        ],
        "fill_values": {
            "destination": ["Tokyo", "Barcelona", "Iceland", "Bali", "Portugal",
                          "New Zealand", "Morocco", "Peru", "Vietnam", "Scotland"],
            "region": ["Southeast Asia", "Europe", "South America", "the Mediterranean"],
            "duration": ["10-day", "2-week", "5-day", "3-week", "long weekend"],
            "month": ["March", "June", "September", "December", "October"],
            "budget": ["3000", "5000", "8000", "2000", "10000"],
            "interests": ["hiking and nature", "food and culture", "history and architecture",
                         "beaches and relaxation", "adventure sports"],
            "cuisine": ["local", "seafood", "street food", "fine dining", "vegetarian"],
            "visited": ["Thailand, Japan, and most of Western Europe",
                       "Mexico and Costa Rica", "a few places in Southeast Asia"],
            "ages": ["6 and 9", "3 and 7", "10 and 13", "8, 11, and 14"],
        },
    },

    "data_analysis": {
        "openers": [
            "I have a dataset with {num_rows} rows and columns for {columns}. I need to find {goal}. Where do I start?",
            "Can you help me write a SQL query to {task}? The tables are:\n\n{schema}",
            "I'm building a dashboard for {audience} and need to visualize {metric}. What chart types would work best?",
            "I ran a {test_type} test and got a p-value of {p_value}. My sample size is {sample_size}. Is this result meaningful?",
            "I need to clean this dataset — it has {issue}. What's the best approach in pandas?",
            "How should I set up an A/B test to measure {metric}? We have about {traffic} users per day.",
        ],
        "followups": [
            "The query is running slow on our {db_type} instance. How can I optimize it?",
            "How do I handle the {num_nulls}% null values in the {column} column?",
            "Can you add a {feature} to that query?",
            "What if I need to join this with another table that has {join_issue}?",
            "How would I automate this to run {frequency}?",
            "Can you show me how to plot this in {library}?",
            "The stakeholders want to see {additional_metric} too. How do I add that?",
            "I'm worried about {data_issue}. How do I validate the data before analysis?",
            "How do I export these results to {format}?",
            "Can you help me write the docstring and comments for this code?",
        ],
        "responses": [
            "Here's how I'd approach this:\n\n```python\nimport pandas as pd\n\n{code}\n```\n\n**What this does:**\n1. {step1}\n2. {step2}\n3. {step3}\n\nExpected output: {expected_output}\n\nFor the null values, I'd recommend {null_strategy} because {reasoning}.",
            "Here's the SQL:\n\n```sql\n{sql_code}\n```\n\n**Performance notes:**\n- {perf_note1}\n- {perf_note2}\n- Estimated execution: {exec_time} on {num_rows} rows\n\nMake sure you have an index on `{index_column}` for best performance.",
            "For {metric} visualization, I'd recommend:\n\n**Primary chart:** {chart_type} — because {reasoning}\n**Supporting chart:** {secondary_chart} — shows {secondary_insight}\n\nHere's the code:\n\n```python\n{viz_code}\n```\n\nKey design decisions: {design_notes}",
            "Let's look at the statistical results:\n\n- **P-value:** {p_value} ({significance})\n- **Effect size:** {effect_size}\n- **Confidence interval:** {ci}\n- **Power:** {power}\n\nInterpretation: {interpretation}\n\n{caveat}",
        ],
        "fill_values": {
            "num_rows": ["50K", "500K", "2M", "10M", "100M"],
            "db_type": ["PostgreSQL", "BigQuery", "Snowflake", "MySQL", "Redshift"],
            "library": ["matplotlib", "seaborn", "plotly", "Altair"],
            "format": ["CSV", "Excel", "a Looker dashboard", "a Slack report"],
            "metric": ["conversion rate", "revenue per user", "churn rate",
                      "session duration", "engagement score"],
            "frequency": ["daily", "hourly", "weekly", "in real-time"],
        },
    },

    "business_strategy": {
        "openers": [
            "We're a {stage} {business_type} with {revenue} in ARR. We're trying to decide between {option_a} and {option_b}. What framework should we use?",
            "I need to build a go-to-market strategy for our new {product_type}. Target market is {market}.",
            "Our {metric} has been declining for {timeframe}. What are the likely causes and how do we diagnose this?",
            "We're considering raising a {round} round. What metrics do investors typically want to see for a {business_type}?",
            "How should we structure pricing for our {product_type}? Current model is {current_pricing} but we're not sure it's optimal.",
        ],
        "followups": [
            "What competitive advantages should we emphasize in our positioning?",
            "How do we measure the success of this strategy? What KPIs should we track?",
            "What's the biggest risk with this approach?",
            "How have similar companies handled this transition?",
            "Can you help me build a financial model for this?",
            "How should we communicate this change to our existing customers?",
            "What's the timeline for seeing results from this strategy?",
            "Should we hire for this internally or bring in consultants?",
            "How do we prioritize when we have limited resources?",
            "What if our competitor launches {counter_move} in response?",
        ],
        "responses": [
            "I'd recommend using the {framework} framework. Here's how to apply it:\n\n**{step1_name}:** {step1_detail}\n**{step2_name}:** {step2_detail}\n**{step3_name}:** {step3_detail}\n\nFor your specific situation ({context}), the key factors are:\n1. {factor1}\n2. {factor2}\n3. {factor3}\n\nBased on this analysis, {recommendation}.",
            "Here's a GTM strategy outline:\n\n**Target Segments (prioritized):**\n1. {segment1} — {segment1_rationale}\n2. {segment2} — {segment2_rationale}\n\n**Channels:**\n{channel_strategy}\n\n**Pricing:** {pricing_recommendation}\n\n**Timeline:**\n- Month 1-2: {phase1}\n- Month 3-4: {phase2}\n- Month 5-6: {phase3}\n\n**Key Metrics:** {metrics}\n\nBiggest risk: {risk}. Mitigation: {mitigation}.",
        ],
        "fill_values": {
            "stage": ["early-stage", "Series A", "Series B", "bootstrapped", "pre-revenue"],
            "business_type": ["SaaS company", "marketplace", "fintech startup", "D2C brand",
                            "B2B platform", "developer tools company"],
            "revenue": ["$500K", "$2M", "$10M", "$50M"],
            "round": ["seed", "Series A", "Series B", "bridge"],
            "metric": ["retention rate", "MRR", "conversion rate", "NPS score"],
            "timeframe": ["the past quarter", "6 months", "a year"],
            "product_type": ["SaaS platform", "API product", "mobile app", "enterprise tool"],
            "market": ["SMBs", "enterprise", "developers", "healthcare providers"],
        },
    },

    "health_fitness": {
        "openers": [
            "I want to start a workout routine but I'm a complete beginner. I have {equipment} and can work out {frequency}. What do you suggest?",
            "I've been lifting for {experience} but I've hit a plateau on my {lift}. Currently at {weight}. How do I break through?",
            "Can you help me build a meal plan? I'm trying to {goal}. I weigh {body_weight} and I'm {height}.",
            "I'm training for a {event} in {timeframe}. Currently I can {current_ability}. What's a good training plan?",
            "I've been having {pain} after {activity}. Is this normal or should I see a doctor?",
            "What's the best way to improve my {fitness_attribute}? I do {current_routine} currently.",
        ],
        "followups": [
            "How many sets and reps should I do for each exercise?",
            "What should I eat before and after the workout?",
            "I don't have access to a {equipment}. What's a good substitute exercise?",
            "How do I know if I'm overtraining?",
            "Can you modify this for someone with {limitation}?",
            "How long before I start seeing results?",
            "Should I take any supplements?",
            "What about rest days? How many do I need?",
            "How do I track my progress effectively?",
            "I find {exercise} really boring. Are there alternatives?",
            "What's the right form for {exercise}? I'm worried about injury.",
            "How does sleep affect my results?",
        ],
        "responses": [
            "Here's a {frequency} workout plan tailored to your level:\n\n**Day 1 — {day1_focus}**\n{day1_exercises}\n\n**Day 2 — {day2_focus}**\n{day2_exercises}\n\n**Day 3 — {day3_focus}**\n{day3_exercises}\n\n**Key Principles:**\n- {principle1}\n- {principle2}\n- {principle3}\n\nStart with these weights and increase by {progression} each week. Rest {rest_time} between sets.",
            "For your {goal} goal, here's a nutrition framework:\n\n**Daily Targets:**\n- Calories: {calories}\n- Protein: {protein}g\n- Carbs: {carbs}g\n- Fat: {fat}g\n\n**Sample Day:**\n- Breakfast: {meal1}\n- Lunch: {meal2}\n- Dinner: {meal3}\n- Snacks: {snacks}\n\nThe most important factor is {key_factor}. Don't overcomplicate it — consistency matters more than perfection.",
            "To break through your {lift} plateau, try:\n\n**Strategy 1: {strategy1}**\n{strategy1_detail}\n\n**Strategy 2: {strategy2}**\n{strategy2_detail}\n\nI'd try {recommendation} first for {timeframe}. If that doesn't work, switch to {alternative}.\n\nAlso check: are you sleeping {sleep_rec} hours? That's often the overlooked factor.",
        ],
        "fill_values": {
            "equipment": ["just a pair of dumbbells", "a full gym", "no equipment (bodyweight only)",
                         "a barbell and pull-up bar", "resistance bands"],
            "frequency": ["3 days/week", "4 days/week", "5 days/week", "6 days/week"],
            "experience": ["a year", "6 months", "3 years", "5+ years"],
            "lift": ["bench press", "squat", "deadlift", "overhead press"],
            "goal": ["lose fat", "build muscle", "maintain weight", "improve endurance",
                    "gain strength", "eat healthier"],
            "event": ["half marathon", "5K", "marathon", "triathlon", "obstacle course race"],
            "fitness_attribute": ["flexibility", "endurance", "core strength", "grip strength",
                                 "cardiovascular fitness", "mobility"],
        },
    },
}

# ---------------------------------------------------------------------------
# Code snippets for coding_help conversations
# ---------------------------------------------------------------------------

CODE_SNIPPETS = {
    "Python": [
        'def process_batch(items):\n    results = []\n    for item in items:\n        data = fetch_data(item["id"])\n        transformed = transform(data)\n        results.append(transformed)\n    return results',
        'class UserService:\n    def __init__(self, db):\n        self.db = db\n\n    async def get_user(self, user_id):\n        user = await self.db.query("SELECT * FROM users WHERE id = %s", user_id)\n        return user[0] if user else None',
        'def merge_configs(base, override):\n    merged = base.copy()\n    for key, value in override.items():\n        if key in merged and isinstance(merged[key], dict):\n            merged[key] = merge_configs(merged[key], value)\n        else:\n            merged[key] = value\n    return merged',
        'from collections import defaultdict\n\ndef find_duplicates(records):\n    seen = defaultdict(list)\n    for i, record in enumerate(records):\n        key = (record["email"], record["name"])\n        seen[key].append(i)\n    return {k: v for k, v in seen.items() if len(v) > 1}',
    ],
    "TypeScript": [
        'async function fetchUserData(userId: string): Promise<User> {\n  const response = await fetch(`/api/users/${userId}`);\n  if (!response.ok) throw new Error(`HTTP ${response.status}`);\n  return response.json();\n}',
        'interface CacheOptions {\n  ttl: number;\n  maxSize: number;\n}\n\nclass LRUCache<K, V> {\n  private map = new Map<K, V>();\n  constructor(private options: CacheOptions) {}\n\n  get(key: K): V | undefined {\n    const value = this.map.get(key);\n    if (value !== undefined) {\n      this.map.delete(key);\n      this.map.set(key, value);\n    }\n    return value;\n  }\n}',
    ],
    "Go": [
        'func processStream(ctx context.Context, ch <-chan Event) error {\n\tfor {\n\t\tselect {\n\t\tcase event, ok := <-ch:\n\t\t\tif !ok { return nil }\n\t\t\tif err := handle(event); err != nil {\n\t\t\t\tlog.Printf("error handling event: %v", err)\n\t\t\t}\n\t\tcase <-ctx.Done():\n\t\t\treturn ctx.Err()\n\t\t}\n\t}\n}',
        'type WorkerPool struct {\n\ttasks   chan func()\n\twg      sync.WaitGroup\n}\n\nfunc NewWorkerPool(size int) *WorkerPool {\n\tp := &WorkerPool{tasks: make(chan func(), 100)}\n\tfor i := 0; i < size; i++ {\n\t\tp.wg.Add(1)\n\t\tgo func() {\n\t\t\tdefer p.wg.Done()\n\t\t\tfor task := range p.tasks {\n\t\t\t\ttask()\n\t\t\t}\n\t\t}()\n\t}\n\treturn p\n}',
    ],
    "Rust": [
        'fn parse_config(path: &Path) -> Result<Config, ConfigError> {\n    let content = fs::read_to_string(path)\n        .map_err(|e| ConfigError::IoError(e))?;\n    let config: Config = toml::from_str(&content)\n        .map_err(|e| ConfigError::ParseError(e))?;\n    config.validate()?;\n    Ok(config)\n}',
    ],
    "Java": [
        'public class RateLimiter {\n    private final int maxRequests;\n    private final Duration window;\n    private final ConcurrentLinkedDeque<Instant> timestamps = new ConcurrentLinkedDeque<>();\n\n    public synchronized boolean tryAcquire() {\n        Instant now = Instant.now();\n        Instant cutoff = now.minus(window);\n        while (!timestamps.isEmpty() && timestamps.peekFirst().isBefore(cutoff)) {\n            timestamps.pollFirst();\n        }\n        if (timestamps.size() < maxRequests) {\n            timestamps.addLast(now);\n            return true;\n        }\n        return false;\n    }\n}',
    ],
}

SQL_SNIPPETS = [
    "SELECT u.id, u.name, COUNT(o.id) as order_count,\n       SUM(o.total) as lifetime_value\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at >= '2024-01-01'\nGROUP BY u.id, u.name\nHAVING COUNT(o.id) > 5\nORDER BY lifetime_value DESC\nLIMIT 100;",
    "WITH monthly_cohorts AS (\n  SELECT user_id,\n         DATE_TRUNC('month', first_purchase) AS cohort_month,\n         DATE_TRUNC('month', purchase_date) AS activity_month\n  FROM purchases\n)\nSELECT cohort_month,\n       activity_month,\n       COUNT(DISTINCT user_id) AS active_users\nFROM monthly_cohorts\nGROUP BY cohort_month, activity_month\nORDER BY cohort_month, activity_month;",
]

CREATIVE_EXCERPTS = [
    "The last train had left hours ago, and Mara sat alone on the platform, watching moths orbit the single surviving streetlight.",
    "He kept the letter in his coat pocket for forty years, reading it so many times the creases wore through.",
    "The city had forgotten the river, but the river remembered everything.",
    "She built the machine not to predict the future, but to listen to the past.",
    "There were seven doors in the hallway, and behind the seventh, the sound of rain that never stopped.",
]

PROBLEM_STATEMENTS = [
    "Find the maximum subarray sum in an array that may contain negative numbers. Input: [-2, 1, -3, 4, -1, 2, 1, -5, 4]",
    "Given a string of parentheses, determine the minimum number of insertions to make them balanced.",
    "A factory produces widgets at rate r(t) = 50 + 10*sin(t). Find the total output over the interval [0, 2*pi].",
    "Prove that the sum of the first n odd numbers equals n^2.",
    "Calculate the eigenvalues of the matrix [[3, 1], [1, 3]] and explain their geometric meaning.",
    "A ball is thrown upward with velocity 20 m/s. Ignoring air resistance, at what time does it reach maximum height?",
]


# ---------------------------------------------------------------------------
# Conversation generator
# ---------------------------------------------------------------------------

class ConversationGenerator:
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
            # Handle special placeholders not in fill_values
            special = {
                "{order_id}": str(self.rng.randint(100000, 999999)),
                "{case_id}": f"CS-{self.rng.randint(10000, 99999)}",
                "{email}": f"user{self.rng.randint(100,999)}@email.com",
                "{username}": f"user_{self.rng.randint(1000,9999)}",
                "{date}": f"{'January February March April May June July August September October November December'.split()[self.rng.randint(0,11)]} {self.rng.randint(1,28)}",
                "{issue}": self.rng.choice(["delayed shipment", "billing discrepancy",
                                            "product quality", "account access"]),
                "{question}": self.rng.choice(["can I get a status update on my previous ticket?",
                                               "what's your return policy for opened items?",
                                               "how do I update my notification preferences?"]),
                "{finding}": self.rng.choice([
                    "the package is currently held at the regional sorting facility",
                    "there was a system error during payment processing",
                    "the item was shipped from our secondary warehouse",
                    "your account was flagged by our automated security system",
                ]),
                "{step1}": self.rng.choice(["Initiate the refund process", "File a replacement request",
                                            "Update the shipping information", "Reset the account flags"]),
                "{step2}": self.rng.choice(["Send confirmation to your email", "Generate a return label",
                                            "Apply the promotional credit", "Notify the warehouse team"]),
                "{step3}": self.rng.choice(["Follow up within 24 hours", "Schedule a callback",
                                            "Monitor the tracking status"]),
                "{action_taken}": self.rng.choice([
                    "processed the refund", "updated the shipping address",
                    "applied the discount code", "unlocked your account",
                    "filed the replacement order"]),
            }
            for key, value in special.items():
                if key in result:
                    result = result.replace(key, value, 1)
            # Handle code-related placeholders
            if "{code_snippet}" in result:
                lang = "Python"
                for l in CODE_SNIPPETS:
                    if l.lower() in result.lower() or "{" + l.lower() + "}" in result.lower():
                        lang = l
                        break
                for l in CODE_SNIPPETS:
                    if "{" + "language}" not in result and l in result:
                        lang = l
                        break
                snippets = CODE_SNIPPETS.get(lang, CODE_SNIPPETS["Python"])
                result = result.replace("{code_snippet}", self.rng.choice(snippets), 1)
            if "{fixed_code}" in result:
                snippets = CODE_SNIPPETS.get("Python", CODE_SNIPPETS["Python"])
                result = result.replace("{fixed_code}", self.rng.choice(snippets), 1)
            if "{edge_case_code}" in result:
                result = result.replace("{edge_case_code}",
                                       'if value is None:\n    raise ValueError("Input cannot be None")', 1)
            if "{schema}" in result:
                result = result.replace("{schema}", self.rng.choice(SQL_SNIPPETS), 1)
            if "{sql_code}" in result:
                result = result.replace("{sql_code}", self.rng.choice(SQL_SNIPPETS), 1)
            if "{viz_code}" in result:
                result = result.replace("{viz_code}",
                    'import matplotlib.pyplot as plt\n\nfig, ax = plt.subplots(figsize=(10, 6))\nax.plot(df["date"], df["value"])\nax.set_title("Trend Over Time")\nplt.show()', 1)
            if "{code}" in result:
                result = result.replace("{code}",
                    'df = pd.read_csv("data.csv")\ndf = df.dropna(subset=["key_column"])\nresult = df.groupby("category").agg({"value": ["mean", "sum", "count"]})\nprint(result)', 1)
            if "{excerpt}" in result:
                result = result.replace("{excerpt}", self.rng.choice(CREATIVE_EXCERPTS), 1)
            if "{revised_excerpt}" in result:
                result = result.replace("{revised_excerpt}", self.rng.choice(CREATIVE_EXCERPTS), 1)
            if "{problem_statement}" in result:
                result = result.replace("{problem_statement}", self.rng.choice(PROBLEM_STATEMENTS), 1)
            # Generic catch-all for remaining unfilled placeholders
            remaining_generic = {
                "{scene_text}": "The fog rolled in thick as wool, blurring the line between the pier and the grey water. Lena pulled her collar tighter and counted the lighthouse flashes — one, two, pause, three — like a heartbeat from somewhere far away.",
                "{creative_piece}": "We measure the weight of days\nin coffee cups and closing doors,\nin the particular silence\nthat follows the word 'stay'\nwhen no one does.",
                "{element}": "the character dynamic",
                "{technique1}": "used shorter sentences to increase tension",
                "{technique2}": "grounded the abstract emotion in a physical detail",
                "{technique3}": "left the key revelation implicit rather than stated",
                "{reasoning}": "it aligns with the overall design goals and constraints",
                "{suggestion}": "consider expanding the sensory details to ground the reader",
                "{scene_description}": "two old friends meet after years of silence",
                "{characters}": "Maya (guarded, dry humor) and Jin (open, earnest)",
                "{event}": "learned the truth about their family",
                "{next_scene}": "the character arrives at the old house",
                "{subplot}": "a hidden letter discovered in the attic",
                "{approach1}": "Use metaphor to externalize the internal state",
                "{approach2}": "Show it through the character's changed behavior",
                "{approach3}": "Reveal it through dialogue subtext",
                "{example1}": "\"The room felt smaller than she remembered, as if the walls had been listening and leaned in.\"",
                "{example2}": "She rearranged the bookshelf twice, then a third time. She never rearranged the bookshelf.",
                "{example3}": "\"I'm fine,\" she said. \"You should try the soup.\"",
                "{recommendation}": "the second option",
                "{effect}": "closure",
                "{simple_explanation}": "it describes how things relate to each other systematically",
                "{analogy}": "a recipe — you have inputs, a process, and an output",
                "{precise_explanation}": "formally, it defines a mapping between sets with specific properties",
                "{example}": "For instance, consider f(x) = 2x + 1. When x=3, f(3) = 7.",
                "{key_insight}": "once you identify the underlying pattern, the solution follows naturally",
                "{step1}": "Identify the relevant variables and constraints",
                "{step2}": "Set up the equation using the given information",
                "{step3}": "Solve systematically, checking each step",
                "{step4}": "Verify the answer by substituting back",
                "{answer}": "42",
                "{concept_a}": "the first concept",
                "{concept_b}": "the second concept",
                "{def_a}": "describes the relationship between inputs",
                "{def_b}": "describes the transformation of outputs",
                "{use_a}": "when analyzing static relationships",
                "{use_b}": "when modeling dynamic processes",
                "{ex_a}": "correlation analysis",
                "{ex_b}": "regression modeling",
                "{mnemonic}": "think 'A for Analysis, B for Building models'",
                "{real_world_example}": "GPS navigation uses this concept: your phone triangulates position using signal timing from multiple satellites.",
                "{applications}": "engineering, data science, and everyday technology",
                "{mini_quiz}": "Quick check: if the input doubles, what happens to the output?",
                "{follow_question}": "it doesn't seem to work in the edge case",
                "{hypothetical}": "we removed that constraint entirely",
                "{step}": "third",
                "{variation}": "different initial conditions",
                "{claim}": "this always converges to a single solution",
                "{diagnosis}": "how the variables interact in the inner loop",
                "{explanation}": "The function mutates shared state without synchronization.",
                "{key_change}": "wrapping the critical section with proper guards",
                "{reason1}": "It separates concerns cleanly",
                "{reason2}": "It's easier to test in isolation",
                "{reason3}": "It scales better as complexity grows",
                "{approach_a}": "Inline everything for performance",
                "{pros_a}": "Faster execution, fewer function calls",
                "{cons_a}": "Harder to maintain and test",
                "{approach_b}": "Extract into composable modules",
                "{pros_b}": "Clean separation, testable, reusable",
                "{cons_b}": "Slight overhead from abstraction",
                "{decision1}": "Used dependency injection for testability",
                "{decision2}": "Kept the interface minimal",
                "{decision3}": "Added validation at the boundary",
                "{benefit}": "clean separation of concerns",
                "{constraint}": "performance within acceptable bounds",
                "{guarantee}": "no data races under concurrent access",
                "{tradeoff}": "~5% additional latency per operation",
                "{edge_case}": "the input is empty or contains duplicates",
                "{action}": "deploy the updated configuration",
                "{task}": "parses nested JSON and flattens it into a table",
                "{feature}": "real-time notifications",
                "{approach}": "polling-based approach",
                "{alt_approach}": "event-driven approach",
                "{requirement}": "handling concurrent writes safely",
                "{entities}": "users, organizations, projects, and permissions",
                "{counter_move}": "an aggressive pricing change",
                "{place_a}": "the city center",
                "{place_b}": "the coastal area",
                "{area}": "the downtown business district",
                "{nearby_place}": "the neighboring region",
                "{pass_type}": "city tourism",
                "{weather_description}": "warm and mostly dry with occasional afternoon showers",
                "{transport}": "Direct flights available from most major hubs (~$400-600 round trip)",
                "{accommodation}": "Boutique hotels in the old town ($80-150/night) or modern apartments ($60-100/night)",
                "{highlights}": "The historic quarter, the central market, the coastal trail, and the sunset viewpoint",
                "{flight_cost}": "500",
                "{hotel_cost}": "120",
                "{food_cost}": "40",
                "{activity_cost}": "25",
                "{total}": "2,500",
                "{day1}": "Arrive, settle in, explore the neighborhood, welcome dinner at a local favorite",
                "{day2}": "Morning walking tour of the historic center, afternoon at the museum, evening food tour",
                "{day3}": "Day trip to the countryside — vineyards, villages, and a scenic hike",
                "{day4}": "Beach morning, afternoon shopping in the artisan quarter, farewell dinner",
                "{tip1}": "Book the walking tour at least 3 days ahead",
                "{tip2}": "Eat where the locals eat — avoid restaurants with photos on the menu",
                "{tip3}": "Get a transit pass on Day 1 — it pays for itself by Day 2",
                "{advance_booking}": "the cooking class and the sunset boat tour",
                "{comparison}": "more affordable and less crowded",
                "{alternative}": "the more popular neighboring destination",
                "{best_time}": "shoulder season (April-May or September-October)",
                "{min_duration}": "5-7 days",
                "{expected_output}": "a DataFrame with aggregated metrics per category",
                "{null_strategy}": "imputation with the median for numeric columns and 'unknown' for categorical",
                "{perf_note1}": "The CTE avoids repeated subquery evaluation",
                "{perf_note2}": "Consider partitioning the table by date if query frequency is high",
                "{exec_time}": "~2-5 seconds",
                "{index_column}": "user_id",
                "{chart_type}": "a line chart with confidence bands",
                "{secondary_chart}": "a heatmap",
                "{secondary_insight}": "the correlation structure between variables",
                "{design_notes}": "Used a clean, minimal style with colorblind-safe palette",
                "{p_value}": "0.032",
                "{significance}": "significant at alpha=0.05",
                "{effect_size}": "Cohen's d = 0.45 (medium)",
                "{ci}": "[0.12, 0.78]",
                "{power}": "0.82",
                "{interpretation}": "There is a statistically significant difference between the groups, with a medium practical effect.",
                "{caveat}": "Note: with your sample size, you have adequate power, but keep in mind that p-values alone don't tell the full story. Always consider the effect size and confidence interval.",
                "{sample_size}": "500",
                "{test_type}": "t",
                "{num_nulls}": "12",
                "{column}": "revenue",
                "{join_issue}": "many-to-many relationships",
                "{data_issue}": "selection bias in the sample",
                "{additional_metric}": "month-over-month growth",
                "{goal}": "identify the top predictors of churn",
                "{columns}": "user_id, signup_date, activity_count, revenue, churn_flag",
                "{audience}": "the executive team",
                "{traffic}": "50,000",
                "{current_pricing}": "flat monthly fee",
                "{framework}": "SWOT + Porter's Five Forces",
                "{step1_name}": "Market Assessment",
                "{step1_detail}": "Evaluate total addressable market and current penetration",
                "{step2_name}": "Competitive Analysis",
                "{step2_detail}": "Map competitor positioning, pricing, and feature gaps",
                "{step3_name}": "Strategic Options",
                "{step3_detail}": "Score each option on feasibility, impact, and resource requirements",
                "{context}": "your current stage and market position",
                "{factor1}": "Cash runway and burn rate constraints",
                "{factor2}": "Current product-market fit signals",
                "{factor3}": "Competitive dynamics in your segment",
                "{segment1}": "Mid-market SaaS companies (50-500 employees)",
                "{segment1_rationale}": "Highest intent signals, shortest sales cycle",
                "{segment2}": "Enterprise early adopters",
                "{segment2_rationale}": "Higher ACV, strong expansion potential",
                "{channel_strategy}": "Content-led growth + targeted outbound to ICP accounts",
                "{pricing_recommendation}": "Usage-based with committed tiers for predictability",
                "{phase1}": "Foundation — messaging, landing page, initial outreach",
                "{phase2}": "Validation — run campaigns, gather feedback, iterate",
                "{phase3}": "Scale — double down on winning channels",
                "{metrics}": "Pipeline generated, CAC, conversion rate, time-to-close",
                "{risk}": "Slow enterprise sales cycle extends runway pressure",
                "{mitigation}": "Run a parallel self-serve motion for faster feedback loops",
                "{option_a}": "expanding horizontally into adjacent markets",
                "{option_b}": "deepening our current vertical",
                "{day1_focus}": "Upper Body Push",
                "{day1_exercises}": "- Bench Press: 4x8\n- Overhead Press: 3x10\n- Incline Dumbbell Press: 3x12\n- Tricep Dips: 3x15\n- Lateral Raises: 3x15",
                "{day2_focus}": "Lower Body",
                "{day2_exercises}": "- Squats: 4x8\n- Romanian Deadlifts: 3x10\n- Leg Press: 3x12\n- Walking Lunges: 3x12/leg\n- Calf Raises: 4x15",
                "{day3_focus}": "Upper Body Pull",
                "{day3_exercises}": "- Barbell Rows: 4x8\n- Pull-ups: 3xAMRAP\n- Seated Cable Row: 3x12\n- Face Pulls: 3x15\n- Bicep Curls: 3x12",
                "{principle1}": "Progressive overload — increase weight or reps each week",
                "{principle2}": "Form first — never sacrifice technique for heavier weight",
                "{principle3}": "Recovery matters — sleep 7-9 hours and eat enough protein",
                "{progression}": "5 lbs for upper body, 10 lbs for lower body",
                "{rest_time}": "60-90 seconds",
                "{calories}": "2,200",
                "{protein}": "165",
                "{carbs}": "250",
                "{fat}": "70",
                "{meal1}": "Greek yogurt with berries and granola (400 cal)",
                "{meal2}": "Grilled chicken bowl with rice, black beans, and vegetables (600 cal)",
                "{meal3}": "Salmon with roasted sweet potatoes and steamed broccoli (550 cal)",
                "{snacks}": "Protein shake + banana (300 cal), trail mix (200 cal)",
                "{key_factor}": "hitting your protein target consistently",
                "{strategy1}": "Periodization — change rep scheme",
                "{strategy1_detail}": "Spend 3 weeks at 5x5 heavy, then 3 weeks at 4x10 moderate. The variation breaks adaptation.",
                "{strategy2}": "Accessory work — target weak points",
                "{strategy2_detail}": "Add pause reps and tempo work to build strength in the sticking point.",
                "{sleep_rec}": "7-9",
                "{body_weight}": "180 lbs",
                "{height}": "5'10\"",
                "{current_ability}": "run 3 miles at a 9:30/mile pace",
                "{pain}": "knee discomfort",
                "{activity}": "running",
                "{current_routine}": "jogging 3x/week",
                "{limitation}": "a lower back issue",
                "{exercise}": "running",
            }
            for key, value in remaining_generic.items():
                if key in result:
                    result = result.replace(key, value, 1)
            # Break if no more placeholders to fill
            if "{" not in result:
                break
        # Final cleanup: remove any remaining unfilled placeholders
        import re
        result = re.sub(r'\{[a-zA-Z_]+\}', '', result)
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
            # Extend with additional detail
            extensions = [
                "\n\nAdditionally, keep in mind that this approach scales well as your needs grow.",
                "\n\nOne more thing to consider: make sure to test this thoroughly with edge cases before deploying.",
                "\n\nI'd also recommend documenting this decision for future reference, so the team understands the rationale.",
                "\n\nLet me know if you'd like me to elaborate on any of these points or if you have follow-up questions.",
                "\n\nFor context, this is a common pattern that has worked well in similar situations. The key is consistency in execution.",
                "\n\nIt's worth noting that the initial setup might take some effort, but the long-term benefits significantly outweigh the upfront cost.",
                "\n\nFrom my experience, the most common pitfall here is rushing the implementation without proper planning. Take the time to get the foundation right.",
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
        """Generate a single multi-turn conversation."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-turn chat dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--num", type=int, default=None, help="Override number of conversations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", default=None, help="Override output path")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config["dataset"]["seed"]
    generator = ConversationGenerator(config, seed=seed)

    print(f"Generating conversations (seed={seed})...")
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

    # Write Parquet
    output_dir = Path(args.output).parent if args.output else Path(__file__).parent / config["dataset"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / config["dataset"]["output_filename"]

    df.to_parquet(output_path, engine="pyarrow", index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDataset written to: {output_path} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
