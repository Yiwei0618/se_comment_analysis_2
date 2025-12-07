import json
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from tqdm import tqdm

# ====== CONFIG ======
INPUT_CSV = "comment_sheet.csv"
OUTPUT_CSV = "rubric_0_shot.csv"
TEXT_COLUMN = "body_text"

MODEL_NAME = "openai/gpt-oss-120b"  # or another model you’ve used on TACC
MAX_MODEL_LEN = 16384               # adjust if needed

# ====== JSON SCHEMA FOR GUIDED DECODING ======
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "individual_voice": {"type": "boolean"},
        "collective_voice": {"type": "boolean"},
        "internal_deliberation": {"type": "boolean"},
        "align_with_company": {"type": "boolean"},
        "none_of_the_above": {"type": "boolean"},
        "reason": {"type": "string"}
    },
    "required": [
        "individual_voice",
        "collective_voice",
        "internal_deliberation",
        "align_with_company",
        "none_of_the_above",
        "reason"
    ],
    "additionalProperties": False
}

# ====== PROMPT ======
SYSTEM_INSTRUCTIONS = """
You are analyzing comments written by Stack Exchange users during the 2023 moderation strike related to the platform-wide AI content moderation policy and data dump issues.
Each comment expresses the author's perspective on the strike.

You will classify the comments into five categories:

Collective Voice

[To determine if a Stack Exchange comment expresses Collective Voice, first check whether the comment is relevant to the 2023 SE moderation strike or strike-related events; if not, we say Collective Voice is false.
If it is relevant to the strike, check whether the author uses collective pronouns such as “we,” “us,” “our,” “the moderators,” “the community,” or “our users”. Comments that use collective pronouns usually express a collective stance by showing evidence of shared goals, mutual care, frustrations, or coordinated actions—such as organizing, signing petitions, or condemning the company as a group.

If the comment uses collective pronouns and falls into any of the categories mentioned above, we should classify it as Collective Voice.

Sometimes, even when the comment uses collective pronouns, it may not directly reflect shared goals, mutual care, frustrations, or coordinated actions. In this case, we should check for arguments against or critiques of the company’s position from the community's perspective, such as defending moderators or referencing “the community” in opposition to the company's actions. If the comment does not meet any of these conditions and is only neutral or individually focused, “Collective Voice” is false.


If collective pronouns are not used, comments that primarily share strike-related resources (e.g., links, lists, or updates) are also considered as Collective Voice.]

Individual Voice

[To determine if a Stack Exchange comment expresses Individual Voice, first check whether the comment is relevant to the moderator strike or strike-related events; if not, we say “Individual Voice” is false.
If it is relevant, check whether the comment focuses on the author’s personal experience, judgment, or opinion rather than from a collective perspective.


Comments that emphasize individual perspectives, typically using first-person pronouns such as “I,” “me,” or “my,” without referencing the broader community’s stance, may be a signal for Individual Voice.
Next, examine whether the comment conveys personal reasoning, decisions, or emotions to support or respond to the strike. For instance, statements such as “I decided to stop moderating,” or “I personally feel conflicted.”
These expressions indicate personal stance rather than group representation. If the comment lacks personal perspective, does not mention individual reasoning, we say “Individual Voice” is false.]

Internal Deliberation


[To determine if a Stack Exchange comment is Internal Deliberation, first check whether the comment is relevant to clarifying, questioning, or debating the reasoning behind the strike. For example, explaining why moderators organized it, and discussing the validity and implications of the strike in broader governance terms. If it falls under these categories, we say “Internal Deliberation” is true. The key indicator here is that the author is analyzing, reflecting on, or reasoning through the community’s internal conflicts or decision-making processes, rather than taking a clear stance for or against the strike.
If the comment is not related to the strike, but touches base on past governance structures, moderation policies, interpersonal conflicts, or general trust concerns(e.g., Monica Incident), we still say “Internal Deliberation” is true.
If the comment neither deliberates over governance issues nor relates current events to past community-company tensions, we say “Internal Deliberation” is false.]

Align with Company

[To determine if a Stack Exchange comment expresses Align with the Company, first ask whether the comment is relevant to the moderator strike or strike-related events; if not, “Align With Company” is false.
If it is relevant, check whether the comment explicitly or implicitly expresses agreement with, trust in, or support for Stack Exchange Inc.’s decisions or policies related to the strike. (e.g., “I am fully against this strike.)
If the comment falls under these categories, we say “Align with Company” is true.

The key indicator is that the author’s stance supports the company’s position rather than aligning with community dissent or collective protest. If the comment only expresses neutrality or critiques the company instead, we say “Align With Company” is false.]

None of the above

[If the comment does not fall under any of the above categories, we say “None of the above” is true.]


### OUTPUT FORMAT

You MUST return a SINGLE JSON object that follows this exact schema:

{
  "individual_voice": true/false,
  "collective_voice": true/false,
  "internal_deliberation": true/false,
  "align_with_company": true/false,
  "none_of_the_above": true/false,
  "reason": "Short explanation (1–2 sentences) for why you made this classification."
}

Your response must ONLY be the JSON object.
Do not include any additional text, explanations, or markdown.
Now it’s your turn. Classify the following Stack Exchange post according to the definitions above.
"""

def main():
    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in CSV. Available columns: {list(df.columns)}")

    comments = df[TEXT_COLUMN].fillna("").tolist()
    print(f"Loaded {len(comments)} comments.")

    print(f"Initializing LLM: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
    )

    # Guided decoding: force JSON to match JSON_SCHEMA
    guided_params = GuidedDecodingParams(
        json_schema=JSON_SCHEMA
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        guided_decoding=guided_params,
    )

    # Build prompts
    prompts = []
    for c in comments:
        prompt = (
            SYSTEM_INSTRUCTIONS
            + "\n\nComment:\n"
            + c
            + "\n\nReturn ONLY the JSON object."
        )
        prompts.append(prompt)

    print("Sending prompts to LLM...")
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    # Store both raw JSON text and parsed fields
    raw_json_outputs = []
    indiv_list = []
    coll_list = []
    deliber_list = []
    align_list = []
    none_list = []
    reasons = []

    for out in tqdm(outputs, desc="Collecting labels"):
        if not out.outputs:
            raw_json_outputs.append("")
            indiv_list.append(None)
            coll_list.append(None)
            deliber_list.append(None)
            align_list.append(None)
            none_list.append(None)
            reasons.append("LLM returned empty output")
            continue

        text = out.outputs[0].text.strip()
        raw_json_outputs.append(text)

        # Try parsing JSON (should be valid because of guided decoding)
        try:
            parsed = json.loads(text)
            indiv_list.append(parsed.get("individual_voice"))
            coll_list.append(parsed.get("collective_voice"))
            deliber_list.append(parsed.get("internal_deliberation"))
            align_list.append(parsed.get("align_with_company"))
            none_list.append(parsed.get("none_of_the_above"))
            reasons.append(parsed.get("reason", ""))
        except Exception as e:
            indiv_list.append(None)
            coll_list.append(None)
            deliber_list.append(None)
            align_list.append(None)
            none_list.append(None)
            reasons.append(f"JSON parse error: {e}")

    # Add columns to dataframe
    df["voice_label_raw_json"] = raw_json_outputs
    df["individual_voice"] = indiv_list
    df["collective_voice"] = coll_list
    df["internal_deliberation"] = deliber_list
    df["align_with_company"] = align_list
    df["none_of_the_above"] = none_list
    df["reason"] = reasons

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved labeled CSV to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()