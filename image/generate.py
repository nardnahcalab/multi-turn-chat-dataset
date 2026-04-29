#!/usr/bin/env python3
"""Generate synthetic multi-turn image chat dataset using Wikipedia images.

Fetches images from Wikipedia articles across diverse topics, then generates
synthetic multi-turn conversations where the first user message references
the image URL in multimodal format. Subsequent turns are text-only follow-ups
about the image.

Outputs:
  - Parquet file with full conversation metadata
  - aiperf multi_turn JSONL (one session per line, user turns only)
  - aiperf mooncake_trace JSONL (one turn per line, full message context)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import uuid
from pathlib import Path

import pandas as pd
import requests
import sys
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
# Wikipedia image fetching
# ---------------------------------------------------------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "MultiTurnChatDatasetGenerator/1.0 (https://github.com/nardnahcalab/multi-turn-chat-dataset)"


def fetch_wikipedia_images(config: dict, cache_path: Path) -> list[dict]:
    """Fetch images from Wikipedia articles and cache locally."""
    target_count = config["images"]["count"]
    min_w = config["images"].get("min_width", 400)
    min_h = config["images"].get("min_height", 300)

    # Check cache
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) >= target_count:
            print(f"Using {target_count} cached images from {cache_path}")
            return cached[:target_count]
        print(f"Cache has {len(cached)} images, need {target_count}. Fetching more...")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    seen_urls: set[str] = set()

    # Fetch from ALL topics to ensure diversity, then balance
    topics = config["images"]["topics"]
    per_topic_images: dict[str, list[dict]] = {}

    for topic_cfg in topics:
        category = topic_cfg["category"]
        articles = topic_cfg["articles"]
        topic_imgs: list[dict] = []
        print(f"  Fetching images for topic: {category} ({len(articles)} articles)")

        for article in articles:
            images = _fetch_article_images(session, article, min_w, min_h)
            for img in images:
                if img["url"] not in seen_urls:
                    img["topic"] = category
                    img["source_article"] = article
                    topic_imgs.append(img)
                    seen_urls.add(img["url"])

            # Polite delay between articles
            time.sleep(0.5)

        per_topic_images[category] = topic_imgs
        print(f"    -> {len(topic_imgs)} images")

    # Balance across topics: take equal share from each, fill remainder round-robin
    all_images: list[dict] = []
    num_topics = len(per_topic_images)
    per_topic_target = target_count // num_topics

    remaining: list[dict] = []
    for category, imgs in per_topic_images.items():
        all_images.extend(imgs[:per_topic_target])
        remaining.extend(imgs[per_topic_target:])

    # Fill remaining slots round-robin from overflow
    shortfall = target_count - len(all_images)
    if shortfall > 0 and remaining:
        all_images.extend(remaining[:shortfall])

    print(f"  Fetched {sum(len(v) for v in per_topic_images.values())} unique images, using {len(all_images)}")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_images, f, indent=2)

    return all_images[:target_count]


def _fetch_article_images(
    session: requests.Session,
    article_title: str,
    min_width: int = 400,
    min_height: int = 300,
) -> list[dict]:
    """Fetch images embedded in a Wikipedia article with metadata."""
    # Step 1: Get list of image titles from article
    params = {
        "action": "query",
        "titles": article_title,
        "prop": "images",
        "imlimit": 50,
        "format": "json",
    }
    try:
        resp = session.get(WIKI_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"    Warning: Failed to list images for '{article_title}': {e}")
        return []

    pages = data.get("query", {}).get("pages", {})
    image_titles: list[str] = []
    for page_data in pages.values():
        for img in page_data.get("images", []):
            title = img["title"]
            # Filter out icons, logos, SVGs, and non-photo files
            lower = title.lower()
            if any(ext in lower for ext in (".svg", ".gif", ".tif")):
                continue
            if any(skip in lower for skip in (
                "icon", "logo", "flag", "symbol", "button",
                "arrow", "commons-logo", "wiki", "edit-clear",
                "padlock", "ambox", "disambiguation", "stub",
                "question_book", "folder_hexagonal", "text-x",
                "increase2", "decrease2", "steady2",
            )):
                continue
            image_titles.append(title)

    if not image_titles:
        return []

    # Step 2: Get image info (URLs, dimensions, metadata)
    results: list[dict] = []
    for i in range(0, len(image_titles), 50):
        batch = image_titles[i:i + 50]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "format": "json",
        }
        try:
            resp = session.get(WIKI_API, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            continue

        for page_data in data.get("query", {}).get("pages", {}).values():
            if "imageinfo" not in page_data:
                continue
            info = page_data["imageinfo"][0]

            width = info.get("width", 0)
            height = info.get("height", 0)
            mime = info.get("mime", "")

            # Filter: minimum size, only raster images
            if width < min_width or height < min_height:
                continue
            if not mime.startswith("image/"):
                continue
            if mime in ("image/svg+xml", "image/gif"):
                continue

            ext_meta = info.get("extmetadata", {})
            description_raw = ext_meta.get("ImageDescription", {}).get("value", "")
            # Strip HTML tags from description
            description = re.sub(r"<[^>]+>", "", description_raw).strip()

            artist_raw = ext_meta.get("Artist", {}).get("value", "")
            artist = re.sub(r"<[^>]+>", "", artist_raw).strip()

            results.append({
                "title": page_data.get("title", "").replace("File:", ""),
                "url": info["url"],
                "width": width,
                "height": height,
                "mime_type": mime,
                "description": description[:500] if description else "",
                "artist": artist[:200] if artist else "",
                "license": ext_meta.get("LicenseShortName", {}).get("value", ""),
            })

        time.sleep(0.3)

    return results


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert visual analyst and image interpreter. You help users "
    "understand, analyze, and discuss images in detail. You can describe visual "
    "elements, analyze composition and techniques, provide historical and cultural "
    "context, and offer creative interpretations. When discussing images, be "
    "specific about what you observe versus your interpretations."
)

# ---------------------------------------------------------------------------
# Conversation templates
# ---------------------------------------------------------------------------

CONVERSATION_TEMPLATES = {
    "image_description": {
        "openers": [
            "Can you describe what you see in this image? I'd like a detailed breakdown of all the visual elements.",
            "What's in this image? Please describe it thoroughly, including the main subject and background details.",
            "I found this interesting image. Can you tell me everything you observe in it?",
            "Please provide a comprehensive description of this image, noting all significant elements and their arrangement.",
            "Walk me through what this image shows, from the most prominent features to the subtle details.",
        ],
        "followups": [
            "Can you be more specific about the {visual_element} in the image?",
            "What about the colors? How would you describe the color palette used here?",
            "Are there any details in the background I might have missed?",
            "How would you describe the overall mood or atmosphere of this image?",
            "Can you describe the spatial relationships between the different elements?",
            "What textures or patterns do you notice?",
            "If you had to describe this image to someone who couldn't see it, what would you emphasize?",
            "What's the most striking or unusual feature you notice?",
            "How does the {visual_element} relate to the rest of the composition?",
            "Can you describe the lighting conditions in the image?",
            "What sense of scale does the image convey?",
            "Are there any {visual_element} details that suggest when or where this was taken?",
        ],
        "responses": [
            "Looking at this image, I can see {description_element}. The composition features {composition_detail}, with {color_detail} tones creating a {mood} atmosphere.",
            "This image shows {description_element}. In the foreground, {foreground_detail}, while the background reveals {background_detail}. The overall scene conveys a sense of {mood}.",
            "The image presents {description_element} with notable {visual_feature}. The {visual_element} stands out due to its {quality}, and the {composition_detail} draws the eye naturally across the frame.",
            "Here's what I observe: {description_element}. The {color_detail} palette works well with the {composition_detail}. {additional_observation}.",
        ],
    },
    "visual_analysis": {
        "openers": [
            "Can you analyze the visual composition of this image? I'm interested in the artistic and technical aspects.",
            "What compositional techniques can you identify in this image? Please discuss the visual design choices.",
            "I'd like a professional visual analysis of this image. What stands out in terms of composition, color, and form?",
            "From an artistic perspective, how would you analyze the visual elements of this image?",
            "Please break down the visual design of this image. What techniques create its overall impact?",
        ],
        "followups": [
            "How does the rule of thirds or other composition rules apply here?",
            "What about the use of {visual_element} — how does it affect the viewer's experience?",
            "Can you discuss the color theory at work in this image?",
            "How does the lighting contribute to the overall composition?",
            "What creates the sense of depth in this image?",
            "Are there any leading lines or framing techniques being used?",
            "How does the balance of positive and negative space work here?",
            "What role does {visual_element} play in guiding the viewer's eye?",
            "How would you rate the visual hierarchy in this image?",
            "What makes this composition effective or memorable?",
            "Can you identify any symmetry or asymmetry at play?",
            "How does the {visual_element} choice impact the emotional tone?",
        ],
        "responses": [
            "From a compositional standpoint, this image employs {composition_technique}. The {visual_element} creates {visual_effect}, while the {color_detail} palette establishes {mood}.",
            "The visual analysis reveals several interesting choices: {composition_technique} is used to draw attention to {focal_point}. The interplay of {visual_element} and {color_detail} produces {visual_effect}.",
            "Examining the composition, I notice {composition_technique} at work. The {visual_element} serves as {compositional_role}, and the overall {color_detail} scheme reinforces the {mood} quality.",
            "This image demonstrates strong {composition_technique}. The arrangement of {visual_element} creates {visual_effect}, supported by the {color_detail} tones throughout.",
        ],
    },
    "contextual_discussion": {
        "openers": [
            "What's the historical or cultural significance of what's shown in this image? Can you provide some context?",
            "I'd like to understand the context behind this image. What can you tell me about its subject matter?",
            "Can you discuss the background and significance of what we see in this image?",
            "What story does this image tell? Help me understand the broader context of what's depicted.",
            "Please explain the cultural, historical, or scientific importance of the subject shown in this image.",
        ],
        "followups": [
            "How has this subject evolved or changed over time?",
            "What role does {context_element} play in the broader context?",
            "Are there any interesting facts or lesser-known details about this subject?",
            "How does this relate to similar subjects in other {context_domain}?",
            "What impact has this had on {context_domain}?",
            "Can you tell me more about the {context_element} aspect?",
            "Why is this considered significant in the field of {context_domain}?",
            "What controversies or debates surround this subject?",
            "How do different {context_domain} perspectives view this?",
            "What would someone need to know to fully appreciate what's shown here?",
            "How does this connect to broader themes in {context_domain}?",
            "What makes this particular example noteworthy compared to others?",
        ],
        "responses": [
            "The subject shown has significant importance in {context_domain}. Historically, {historical_detail}. The {context_element} aspect is particularly notable because {significance_detail}.",
            "This image captures something with deep {context_domain} roots. {historical_detail}. It represents {significance_detail}, which has influenced {impact_area}.",
            "Understanding the context here is important: {historical_detail}. In the realm of {context_domain}, this subject holds a special place because {significance_detail}.",
            "From a {context_domain} perspective, what we see has a rich background. {historical_detail}. The {context_element} dimension adds another layer of meaning: {significance_detail}.",
        ],
    },
    "creative_interpretation": {
        "openers": [
            "What emotions or stories does this image evoke for you? I'd love a creative interpretation.",
            "If you were writing a story inspired by this image, what narrative would you create?",
            "Can you provide an artistic or poetic interpretation of this image? What does it make you think about?",
            "What deeper meanings or symbolic interpretations can you find in this image?",
            "How would you interpret this image from a creative or philosophical perspective?",
        ],
        "followups": [
            "What symbolism do you see in the {visual_element}?",
            "How might different people interpret this image differently?",
            "If this were a scene from a movie, what would be happening next?",
            "What metaphor does the {visual_element} suggest to you?",
            "How does the {mood} mood connect to broader human experiences?",
            "Can you write a short poem or prose piece inspired by this image?",
            "What would the {visual_element} represent in a dream?",
            "How might an artist from a different era have depicted this same subject?",
            "What questions does this image raise in your mind?",
            "How does this image connect to the theme of {abstract_concept}?",
            "What contrasts or tensions do you see in the image?",
            "If this image had a soundtrack, what would it sound like?",
        ],
        "responses": [
            "This image evokes a strong sense of {mood}. The {visual_element} could be seen as a metaphor for {abstract_concept}. Looking deeper, {creative_observation}.",
            "Creatively, I see {creative_observation}. The {visual_element} suggests {abstract_concept}, while the overall atmosphere speaks to {mood}. One could interpret this as {interpretation}.",
            "From a creative standpoint, this image resonates with themes of {abstract_concept}. The {visual_element} particularly stands out as {creative_observation}. The {mood} quality invites reflection on {interpretation}.",
            "My interpretation centers on the {visual_element} as a symbol of {abstract_concept}. {creative_observation}. The image as a whole seems to suggest {interpretation}.",
        ],
    },
    "comparison": {
        "openers": [
            "How does the subject in this image compare to similar subjects you know about? What makes it distinctive?",
            "Can you compare what's shown here with other well-known examples of similar subjects?",
            "What sets this apart from other images or depictions of similar subjects?",
            "How does this compare to the most famous or iconic versions of similar subjects?",
            "I'd like to understand how this image's subject relates to and differs from comparable examples.",
        ],
        "followups": [
            "What are the key differences in {visual_element} compared to {comparison_subject}?",
            "How does the {context_element} compare across different examples?",
            "Which version or example do you think is most impressive and why?",
            "What common elements connect this with similar subjects?",
            "How has the depiction of such subjects changed over time?",
            "What unique features distinguish this from its closest comparisons?",
            "Can you compare the {visual_element} approaches used in different examples?",
            "What makes this representation more or less effective than alternatives?",
            "How do cultural differences influence how this subject is depicted?",
            "What trends do you notice when comparing multiple examples?",
            "If you ranked this among similar subjects, where would it fall?",
            "What could we learn by studying the differences between these examples?",
        ],
        "responses": [
            "Comparing this to {comparison_subject}, several differences stand out. The {visual_element} here is distinctive because {distinction_detail}. Unlike {comparison_subject}, this example features {unique_feature}.",
            "In comparison with similar subjects, this stands out through its {visual_element}. While {comparison_subject} tends to emphasize {comparison_detail}, here we see {distinction_detail}.",
            "When placed alongside {comparison_subject}, the differences become clear. The {visual_element} in this image contrasts with {comparison_detail}. What makes this unique is {unique_feature}.",
            "This bears both similarities and differences to {comparison_subject}. They share {shared_feature}, but diverge in {distinction_detail}. The {visual_element} here is particularly distinctive.",
        ],
    },
    "educational": {
        "openers": [
            "I'm studying this subject and found this image. Can you use it as a starting point to teach me about what's shown?",
            "As an educational exercise, what can we learn from examining this image closely?",
            "Can you explain the concepts visible in this image? I'd like to learn more about what I'm seeing.",
            "Help me understand the science, history, or art behind what's depicted in this image.",
            "I want to learn about this subject. Can you use this image to walk me through the key concepts?",
        ],
        "followups": [
            "Can you explain the {context_element} in simpler terms?",
            "What fundamental principles of {context_domain} does this demonstrate?",
            "What should a beginner know about this subject?",
            "Are there common misconceptions about what's shown here?",
            "How is this topic typically taught in {context_domain} courses?",
            "What related concepts should I explore next?",
            "Can you recommend how to learn more about the {context_element}?",
            "What experiments or observations relate to what's shown?",
            "How does this connect to everyday life or practical applications?",
            "What vocabulary or terminology is important for discussing this subject?",
            "Can you quiz me on what we've discussed about this image?",
            "What are the most important takeaways from studying this image?",
        ],
        "responses": [
            "This image provides an excellent opportunity to learn about {context_element}. The key concept here is {educational_concept}. Notice how {observation_detail}, which demonstrates {principle}.",
            "Let's use this image as a learning tool. What you're seeing illustrates {educational_concept}. The {visual_element} here shows {observation_detail}, which is a great example of {principle}.",
            "From an educational perspective, this image demonstrates {educational_concept}. The {context_element} visible here teaches us about {principle}. Specifically, {observation_detail}.",
            "There's a lot to learn from this image. It illustrates {educational_concept} through the visible {visual_element}. The key takeaway is {principle}, as evidenced by {observation_detail}.",
        ],
    },
    "technical_photography": {
        "openers": [
            "From a photography perspective, how would you evaluate the technical quality of this image?",
            "What photographic or artistic techniques were used to capture this image?",
            "Can you discuss the technical aspects of how this image was created? I'm interested in the craft behind it.",
            "As a photographer, what do you notice about the technical execution of this image?",
            "Please analyze the photographic techniques evident in this image.",
        ],
        "followups": [
            "What camera settings do you think were used for this shot?",
            "How does the {visual_element} technique affect the final result?",
            "What post-processing or editing might have been applied?",
            "How would you improve this image from a technical standpoint?",
            "What challenges would the photographer have faced capturing this?",
            "How does the choice of {visual_element} affect the storytelling?",
            "What alternative approaches could achieve a similar effect?",
            "How does the {visual_element} contribute to the image quality?",
            "What equipment or setup would be needed to recreate this shot?",
            "How does the {visual_element} compare to current photography trends?",
            "What makes this technically impressive or noteworthy?",
            "If you were coaching someone to take a similar photo, what would you advise?",
        ],
        "responses": [
            "From a technical standpoint, this image demonstrates {photography_technique}. The {visual_element} suggests {technical_detail}. The {color_detail} rendition indicates {processing_detail}.",
            "Technically, several things stand out: {photography_technique} is evident in the way {visual_element} is rendered. The {technical_detail} adds to the overall quality, and {processing_detail}.",
            "The technical execution here shows {photography_technique}. Looking at the {visual_element}, I can see {technical_detail}. The overall {color_detail} treatment suggests {processing_detail}.",
            "This image showcases {photography_technique} with notable {visual_element} control. The {technical_detail} is well-executed, and the {processing_detail} enhances the final result.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Fill values for template placeholders
# ---------------------------------------------------------------------------

FILL_VALUES = {
    # Visual elements
    "visual_element": [
        "lighting", "color palette", "texture", "contrast", "shadow",
        "foreground detail", "background element", "focal point", "pattern",
        "line work", "perspective", "depth of field", "framing", "symmetry",
        "negative space", "tone", "gradient", "reflection", "silhouette",
    ],
    "description_element": [
        "a richly detailed scene with multiple layers of visual interest",
        "a striking composition that draws the eye immediately to the central subject",
        "an intricate arrangement of natural and structural elements",
        "a vivid display of color and form that captures the essence of the subject",
        "a carefully balanced scene with both prominent features and subtle nuances",
        "a complex interplay of light and shadow across the scene",
        "a sweeping view that encompasses both grand features and fine details",
    ],
    "composition_detail": [
        "strong diagonal lines guiding the eye through the frame",
        "a clear subject-background separation creating visual depth",
        "balanced asymmetry that creates dynamic tension",
        "centered composition that emphasizes the subject's importance",
        "layered elements creating a sense of spatial progression",
        "a triangular arrangement of key visual elements",
        "strategic use of the rule of thirds",
    ],
    "color_detail": [
        "warm golden", "cool blue", "rich saturated", "muted earthy",
        "high-contrast", "pastel", "monochromatic", "complementary",
        "vibrant", "subdued natural", "deep shadow", "luminous",
    ],
    "mood": [
        "serene tranquility", "dramatic intensity", "nostalgic warmth",
        "awe-inspiring grandeur", "intimate contemplation", "dynamic energy",
        "peaceful harmony", "mysterious depth", "vibrant celebration",
        "quiet solitude", "majestic power", "gentle beauty",
    ],
    "foreground_detail": [
        "detailed textures draw the eye in close",
        "well-defined subjects create an immediate focal point",
        "rich detail rewards careful observation",
        "strong visual elements anchor the composition",
        "intricate patterns provide visual interest up close",
    ],
    "background_detail": [
        "a complementary backdrop that adds depth to the scene",
        "softer elements that provide context without competing for attention",
        "subtle details that reward closer inspection",
        "atmospheric perspective creating a sense of distance",
        "harmonious tones that support the overall composition",
    ],
    "visual_feature": [
        "use of natural light", "textural contrast", "color harmony",
        "spatial depth", "tonal range", "compositional balance",
        "visual rhythm", "atmospheric quality", "fine detail",
    ],
    "quality": [
        "sharpness and clarity", "tonal richness", "visual weight",
        "unique character", "striking presence", "subtle elegance",
        "dramatic impact", "delicate beauty", "bold expression",
    ],
    "additional_observation": [
        "The interplay between natural and structured elements adds visual complexity",
        "Small details throughout the scene reward patient observation",
        "The tonal range from highlights to shadows spans a wide gamut",
        "There's a harmonious relationship between all the elements in the frame",
        "The sense of scale is particularly effective in conveying the subject's grandeur",
    ],
    # Composition analysis
    "composition_technique": [
        "rule of thirds placement", "leading line composition",
        "frame-within-a-frame technique", "golden ratio proportions",
        "symmetrical balance", "dynamic asymmetry",
        "foreground interest anchoring", "layered depth staging",
        "minimalist negative space", "radial composition",
    ],
    "visual_effect": [
        "a natural flow that guides the viewer through the scene",
        "a strong sense of three-dimensional depth",
        "visual tension that keeps the composition engaging",
        "a calming equilibrium across the frame",
        "an immersive quality that draws the viewer into the scene",
        "a dynamic energy that conveys movement and life",
    ],
    "focal_point": [
        "the primary subject", "the central element",
        "the most illuminated area", "the area of highest contrast",
        "the point where leading lines converge",
        "the most detailed region of the image",
    ],
    "compositional_role": [
        "an anchoring element that grounds the composition",
        "a visual counterweight to balance the frame",
        "a pathway leading the eye through the scene",
        "a frame-within-a-frame that isolates the subject",
        "a recurring motif that creates visual rhythm",
    ],
    # Context and history
    "context_element": [
        "historical significance", "cultural importance", "scientific relevance",
        "architectural heritage", "natural ecology", "artistic tradition",
        "technological achievement", "social impact", "geographical uniqueness",
        "conservation status", "symbolic meaning", "economic importance",
    ],
    "context_domain": [
        "art history", "natural science", "cultural studies", "geography",
        "architecture", "environmental science", "photography", "technology",
        "anthropology", "world history", "ecology", "visual arts",
    ],
    "historical_detail": [
        "This subject dates back centuries and has witnessed significant historical events",
        "The origins of this can be traced to a period of major cultural transformation",
        "Throughout history, this has served as an important symbol and gathering point",
        "The development of this subject reflects broader patterns in human achievement",
        "This represents a milestone in the evolution of its field",
        "Historical records show this has been a point of fascination for generations",
    ],
    "significance_detail": [
        "it represents a unique convergence of natural and human-made elements",
        "it has influenced countless subsequent works and developments in the field",
        "it serves as a benchmark against which similar subjects are measured",
        "it embodies principles that remain relevant and influential today",
        "it demonstrates techniques and approaches that were revolutionary for their time",
        "it continues to attract scholarly attention and public fascination",
    ],
    "impact_area": [
        "subsequent artistic movements", "modern conservation efforts",
        "contemporary architectural design", "current scientific understanding",
        "tourism and cultural exchange", "educational curricula worldwide",
        "ongoing research and discovery", "popular culture and media",
    ],
    # Creative interpretation
    "abstract_concept": [
        "the passage of time", "the relationship between humanity and nature",
        "the pursuit of beauty", "the tension between order and chaos",
        "the power of perspective", "the dialogue between old and new",
        "the resilience of the natural world", "the human drive to create",
        "the interconnectedness of all things", "the beauty in imperfection",
        "the cycle of renewal", "the search for meaning",
    ],
    "creative_observation": [
        "the interplay of light and shadow tells a story beyond the literal image",
        "there's a tension between stillness and implied movement that creates drama",
        "the relationship between scale and detail invites contemplation",
        "the visual rhythm creates an almost musical quality to the composition",
        "layers of meaning emerge the longer one studies the image",
        "the juxtaposition of elements creates an unexpected narrative",
    ],
    "interpretation": [
        "a meditation on the enduring nature of beauty across time",
        "a commentary on the relationship between observer and observed",
        "an exploration of how context shapes our understanding of what we see",
        "a celebration of the extraordinary found within the ordinary",
        "a reflection on how perspectives shift with changing light and time",
        "a study in how visual elements can convey emotional depth",
    ],
    # Comparison
    "comparison_subject": [
        "other famous examples in this category",
        "similar subjects from different time periods",
        "comparable works by different artists or creators",
        "parallel examples from other cultures or regions",
        "the most iconic versions of similar subjects",
        "modern interpretations of the same theme",
    ],
    "distinction_detail": [
        "the unique treatment of light and atmosphere sets it apart",
        "the level of detail and preservation gives it special character",
        "the particular angle and perspective offer a fresh view",
        "the cultural context adds layers of meaning not found elsewhere",
        "the scale and ambition of the subject distinguish it",
    ],
    "unique_feature": [
        "an unusual combination of elements rarely seen together",
        "a perspective that reveals aspects often overlooked",
        "details that speak to the specific conditions of its creation",
        "a quality of authenticity that resonates with viewers",
        "technical mastery that elevates it above comparable examples",
    ],
    "comparison_detail": [
        "a more formal or structured approach",
        "different material choices and techniques",
        "a focus on different aspects of the subject",
        "a more idealized or romanticized treatment",
        "greater emphasis on symbolic meaning",
    ],
    "shared_feature": [
        "a common commitment to capturing the essence of the subject",
        "similar compositional principles guiding the arrangement",
        "comparable use of natural elements as visual anchors",
        "shared cultural references and visual traditions",
        "a mutual emphasis on visual impact and memorability",
    ],
    # Educational
    "educational_concept": [
        "fundamental principles of visual composition and design",
        "the relationship between form, function, and aesthetics",
        "how environmental factors shape visual outcomes",
        "the interplay of technical skill and artistic vision",
        "core principles that govern how we perceive and interpret visual information",
        "the scientific or historical processes that produced what we see",
    ],
    "observation_detail": [
        "the relationship between different visual elements reveals underlying principles",
        "careful examination shows how multiple factors contribute to the overall effect",
        "the progression from broad features to fine details illustrates important concepts",
        "the way light interacts with surfaces demonstrates key optical principles",
        "the arrangement of elements follows patterns found throughout the natural and designed world",
    ],
    "principle": [
        "how visual perception shapes our understanding of the world",
        "the fundamental rules that govern effective visual communication",
        "the scientific principles underlying natural phenomena",
        "how cultural context influences our interpretation of visual information",
        "the relationship between structure and appearance in both natural and made objects",
    ],
    # Photography technique
    "photography_technique": [
        "careful exposure management", "selective focus and depth control",
        "strategic use of natural lighting", "long exposure technique",
        "high dynamic range processing", "precise white balance calibration",
        "thoughtful aperture selection", "expert timing and anticipation",
        "macro or close-up technique", "wide-angle perspective control",
    ],
    "technical_detail": [
        "well-controlled depth of field that separates subject from background",
        "accurate exposure that preserves detail in both highlights and shadows",
        "sharp focus on the key subject with pleasing background blur",
        "natural-looking color rendition that preserves the scene's character",
        "good noise control even in challenging lighting conditions",
        "precise timing that captures a decisive moment",
    ],
    "processing_detail": [
        "careful post-processing that enhances without appearing artificial",
        "subtle adjustments to tone and contrast that bring out fine detail",
        "color grading that reinforces the intended mood of the image",
        "judicious use of sharpening and noise reduction",
        "well-balanced exposure adjustments across the tonal range",
    ],
}


# ---------------------------------------------------------------------------
# Conversation generator
# ---------------------------------------------------------------------------

class ImageConversationGenerator:
    """Generate synthetic multi-turn conversations about images."""

    def __init__(self, config: dict, images: list[dict], seed: int = 42):
        self.config = config
        self.images = images
        self.rng = random.Random(seed)

        self.conv_types = config["conversation_types"]
        self.conv_weights = [ct["weight"] for ct in self.conv_types]

        self.turn_cfg = config["turns"]
        self.length_cfg = config["response_length"]

    # -- template helpers ---------------------------------------------------

    def _fill_template(self, template: str, image: dict) -> str:
        """Fill placeholders in a template string."""
        # Phase 1: image-specific fills
        image_fills = {
            "{image_title}": image.get("title", "Untitled"),
            "{image_url}": image.get("url", ""),
            "{image_description}": image.get("description", ""),
            "{image_artist}": image.get("artist", "Unknown"),
            "{image_topic}": image.get("topic", ""),
            "{source_article}": image.get("source_article", ""),
        }
        result = template
        for placeholder, value in image_fills.items():
            result = result.replace(placeholder, value)

        # Phase 2: generic fills from FILL_VALUES
        for key, values in FILL_VALUES.items():
            placeholder = "{" + key + "}"
            while placeholder in result:
                result = result.replace(placeholder, self.rng.choice(values), 1)

        # Phase 3: clean up any unfilled placeholders
        result = re.sub(r"\{[a-zA-Z_0-9]+\}", "", result)
        return result.strip()

    def _response_length_bucket(self, turn_index: int) -> str:
        """Pick a response length bucket based on turn position."""
        dist_config = self.length_cfg["length_distribution_by_turn"]
        if turn_index < 3:
            dist = dist_config["early"]
        elif turn_index < 15:
            dist = dist_config["middle"]
        else:
            dist = dist_config["late"]
        buckets = list(dist.keys())
        weights = list(dist.values())
        return self.rng.choices(buckets, weights=weights, k=1)[0]

    def _generate_user_message(self, conv_type: str, image: dict, turn_index: int) -> str:
        """Generate a user message (opener for turn 0, followup otherwise)."""
        templates = CONVERSATION_TEMPLATES[conv_type]
        if turn_index == 0:
            template = self.rng.choice(templates["openers"])
        else:
            template = self.rng.choice(templates["followups"])
        return self._fill_template(template, image)

    def _generate_response(self, conv_type: str, image: dict, turn_index: int) -> str:
        """Generate an assistant response with appropriate length."""
        templates = CONVERSATION_TEMPLATES[conv_type]["responses"]
        base = self._fill_template(self.rng.choice(templates), image)

        bucket = self._response_length_bucket(turn_index)
        length_config = self.length_cfg[bucket]
        target_words = self.rng.randint(length_config["min_words"], length_config["max_words"])

        words = base.split()

        # Pad if too short
        extensions = [
            "\n\nIt's also worth noting that the visual qualities here create layers "
            "of meaning that reward closer inspection.",
            "\n\nFrom a broader perspective, this connects to larger themes in how "
            "we perceive and interpret visual information.",
            "\n\nAnother dimension worth exploring is how the specific conditions "
            "under which this was created shaped the final result.",
            "\n\nThe interplay of elements here demonstrates principles that appear "
            "across many different visual contexts.",
            "\n\nLooking more carefully, subtle details emerge that add richness to "
            "the overall visual experience.",
            "\n\nThis level of quality is achieved through careful attention to both "
            "the broad composition and the fine details.",
            "\n\nWhat makes this particularly interesting is how different viewers "
            "may focus on entirely different aspects of the image.",
            "\n\nThe technical execution complements the artistic vision, creating a "
            "result that works on multiple levels simultaneously.",
            "\n\nConsidering the context in which this exists, additional layers of "
            "significance become apparent to the informed observer.",
            "\n\nThe relationship between the visual elements tells a story that "
            "goes beyond what a simple description can capture.",
        ]
        while len(words) < target_words:
            ext = self._fill_template(self.rng.choice(extensions), image)
            words.extend(ext.split())

        # Trim if too long
        if len(words) > int(target_words * 1.3):
            words = words[:target_words]
            text = " ".join(words)
            last_period = text.rfind(".")
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]
            return text

        return " ".join(words)

    # -- single conversation -----------------------------------------------

    def generate_conversation(self, num_turns: int) -> dict:
        """Generate one multi-turn conversation about an image."""
        # Select conversation type (weighted)
        conv_type_cfg = self.rng.choices(
            self.conv_types, weights=self.conv_weights, k=1
        )[0]
        conv_type = conv_type_cfg["name"]

        # Select image
        image = self.rng.choice(self.images)

        # System message
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # First turn: multimodal (text + image_url)
        first_user_text = self._generate_user_message(conv_type, image, 0)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": first_user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image["url"],
                    },
                },
            ],
        })
        first_response = self._generate_response(conv_type, image, 0)
        messages.append({"role": "assistant", "content": first_response})

        # Track cumulative character lengths
        cumulative_char_lengths: list[int] = []
        running_chars = sum(len(str(m.get("content", ""))) for m in messages)
        cumulative_char_lengths.append(running_chars)

        # Subsequent turns: text-only
        for turn_idx in range(1, num_turns):
            user_msg = self._generate_user_message(conv_type, image, turn_idx)
            messages.append({"role": "user", "content": user_msg})
            running_chars += len(user_msg)

            assistant_msg = self._generate_response(conv_type, image, turn_idx)
            messages.append({"role": "assistant", "content": assistant_msg})
            running_chars += len(assistant_msg)

            cumulative_char_lengths.append(running_chars)

        return {
            "conversation_id": str(uuid.UUID(int=self.rng.getrandbits(128), version=4)),
            "conversation_type": conv_type,
            "image_title": image.get("title", ""),
            "image_url": image["url"],
            "image_topic": image.get("topic", ""),
            "source_article": image.get("source_article", ""),
            "image_width": image.get("width", 0),
            "image_height": image.get("height", 0),
            "num_turns": num_turns,
            "num_messages": len(messages),
            "system_prompt": SYSTEM_PROMPT,
            "messages": json.dumps(messages),
            "total_characters": running_chars,
            "estimated_tokens": running_chars // 4,
            "cumulative_char_lengths": json.dumps(cumulative_char_lengths),
        }

    # -- dataset generation ------------------------------------------------

    def generate_dataset(self, num_conversations: int | None = None) -> list[dict]:
        """Generate the full dataset of conversations."""
        turn_min = self.turn_cfg["min"]
        turn_max = self.turn_cfg["max"]

        conversations: list[dict] = []

        if num_conversations is not None:
            # Override mode: uniform random turns
            for _ in range(num_conversations):
                n_turns = self.rng.randint(turn_min, turn_max)
                conversations.append(self.generate_conversation(n_turns))
        else:
            # Use configured distribution buckets
            dist = self.turn_cfg["distribution"]
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

    For image conversations, the first turn includes the image URL in
    the multimodal content format. Subsequent turns are text-only follow-ups.
    """
    entries = []
    for conv in conversations:
        messages = json.loads(conv["messages"])
        turns = []
        for msg in messages:
            if msg["role"] == "user":
                # Handle multimodal first message
                if isinstance(msg["content"], list):
                    text_parts = []
                    images = []
                    for part in msg["content"]:
                        if part.get("type") == "text":
                            text_parts.append(part["text"])
                        elif part.get("type") == "image_url":
                            images.append(part["image_url"]["url"])
                    text = " ".join(text_parts)
                    if images:
                        text += "\n\n[Image: " + images[0] + "]"
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
    parser = argparse.ArgumentParser(description="Generate synthetic multi-turn image chat dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--num", type=int, default=None, help="Override number of conversations")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--format", choices=["all", "parquet", "aiperf", "mooncake"],
                        default="all", help="Output format(s)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetching images, use cached data")
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

    # Fetch or load cached images
    cache_path = Path(__file__).parent / config["images"]["cache_file"]
    if args.skip_fetch and cache_path.exists():
        print(f"Loading cached images from {cache_path}")
        with open(cache_path) as f:
            images = json.load(f)
    else:
        images = fetch_wikipedia_images(config, cache_path)

    generator = ImageConversationGenerator(config, images, seed=seed)

    num_conversations = args.num
    print(f"Generating image conversations (seed={seed})...")
    conversations = generator.generate_dataset(num_conversations=num_conversations)
    print(f"Generated {len(conversations)} conversations")

    df = pd.DataFrame(conversations)

    print(f"\n--- Dataset Summary ---")
    print(f"Total conversations: {len(df)}")
    print(f"Turn count range: {df['num_turns'].min()} - {df['num_turns'].max()}")
    print(f"Mean turns: {df['num_turns'].mean():.1f}")
    print(f"Conversation type distribution:")
    for ctype, count in df["conversation_type"].value_counts().items():
        print(f"  {ctype}: {count} ({100*count/len(df):.1f}%)")
    print(f"Estimated total tokens: {df['estimated_tokens'].sum():,}")

    output_dir = Path(args.output).parent if args.output else Path(__file__).parent / config["dataset"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_count = len(df)
    descriptive_name = build_descriptive_name(
        config, actual_count, seed, "image", custom_suffix=args.name
    )

    if args.descriptive_names:
        file_base = descriptive_name
    else:
        file_base = config["dataset"]["output_filename"].replace(".parquet", "")

    fmt = args.format
    output_files = {}

    if fmt in ("all", "parquet"):
        parquet_path = Path(args.output) if args.output and fmt == "parquet" else output_dir / f"{file_base}.parquet"
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        output_files["parquet"] = str(parquet_path)
        print(f"\nParquet written to: {parquet_path} ({file_size_mb:.2f} MB)")

    if fmt in ("all", "aiperf"):
        aiperf_entries = convert_to_aiperf_multi_turn(conversations)
        jsonl_path = output_dir / f"{file_base}.jsonl"
        with open(jsonl_path, "w") as f:
            for entry in aiperf_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_size_mb = jsonl_path.stat().st_size / (1024 * 1024)
        output_files["aiperf_multi_turn"] = str(jsonl_path)
        print(f"aiperf multi_turn JSONL written to: {jsonl_path} ({file_size_mb:.2f} MB)")

    if fmt in ("all", "mooncake"):
        mooncake_entries = convert_to_aiperf_mooncake(conversations)
        mooncake_path = output_dir / f"{file_base}_mooncake.jsonl"
        with open(mooncake_path, "w") as f:
            for entry in mooncake_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file_size_mb = mooncake_path.stat().st_size / (1024 * 1024)
        output_files["mooncake_trace"] = str(mooncake_path)
        print(f"aiperf mooncake_trace JSONL written to: {mooncake_path} ({file_size_mb:.2f} MB)")

    if not args.no_profile:
        manifest = build_manifest(
            df=df,
            config=config,
            dataset_type="image",
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
