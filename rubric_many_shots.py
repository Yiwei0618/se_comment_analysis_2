import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ====== CONFIG ======
INPUT_CSV = "comment_sheet.csv"
OUTPUT_CSV = "rubric_many_shots.csv"
TEXT_COLUMN = "body_text"

MODEL_NAME = "openai/gpt-oss-120b"  # or another model you’ve used on TACC
MAX_MODEL_LEN = 16384               # adjust if needed

# ====== PROMPT ======
SYSTEM_INSTRUCTIONS = """
You are analyzing comments written by Stack Exchange users during the 2023 moderation strike related to the platform-wide AI content moderation policy and data dump issues.

Each comment expresses the author's perspective on the strike.

You will classify the comments into five categories:


**Collective Voice**

To determine if a Stack Exchange comment expresses Collective Voice, first check whether the comment is relevant to the 2023 SE moderation strike or strike-related events; if not, we say Collective Voice is false.
If it is relevant to the strike, check whether the author uses collective pronouns such as “we,” “us,” “our,” “the moderators,” “the community,” or “our users”. Comments that use collective pronouns usually express a collective stance by showing evidence of shared goals, mutual care, frustrations, or coordinated actions—such as organizing, signing petitions, or condemning the company as a group.

If the comment uses collective pronouns and falls into any of the categories mentioned above, we should classify it as Collective Voice.

Sometimes, even when the comment uses collective pronouns, it may not directly reflect shared goals, mutual care, frustrations, or coordinated actions. In this case, we should check for arguments against or critiques of the company’s position from the community's perspective, such as defending moderators or referencing “the community” in opposition to the company's actions. If the comment does not meet any of these conditions and is only neutral or individually focused, “Collective Voice” is false.


If collective pronouns are not used, comments that primarily share strike-related resources (e.g., links, lists, or updates) are also considered as Collective Voice.



**Individual Voice**

To determine if a Stack Exchange comment expresses Individual Voice, first check whether the comment is relevant to the moderator strike or strike-related events; if not, we say “Individual Voice” is false.
If it is relevant, check whether the comment focuses on the author’s personal experience, judgment, or opinion rather than from a collective perspective.


Comments that emphasize individual perspectives, typically using first-person pronouns such as “I,” “me,” or “my,” without referencing the broader community’s stance, may be a signal for Individual Voice.
Next, examine whether the comment conveys personal reasoning, decisions, or emotions to support or respond to the strike. For instance, statements such as “I decided to stop moderating,” or “I personally feel conflicted.”
These expressions indicate personal stance rather than group representation. If the comment lacks personal perspective, does not mention individual reasoning, we say “Individual Voice” is false.


**Internal Deliberation**


To determine if a Stack Exchange comment is Internal Deliberation, first check whether the comment is relevant to clarifying, questioning, or debating the reasoning behind the strike. For example, explaining why moderators organized it, and discussing the validity and implications of the strike in broader governance terms. If it falls under these categories, we say “Internal Deliberation” is true. The key indicator here is that the author is analyzing, reflecting on, or reasoning through the community’s internal conflicts or decision-making processes, rather than taking a clear stance for or against the strike.
If the comment is not related to the strike, but touches base on past governance structures, moderation policies, interpersonal conflicts, or general trust concerns(e.g., Monica Incident), we still say “Internal Deliberation” is true.
If the comment neither deliberates over governance issues nor relates current events to past community-company tensions, we say “Internal Deliberation” is false.


**Align with Company**

To determine if a Stack Exchange comment expresses Align with the Company, first ask whether the comment is relevant to the moderator strike or strike-related events; if not, “Align With Company” is false.
If it is relevant, check whether the comment explicitly or implicitly expresses agreement with, trust in, or support for Stack Exchange Inc.’s decisions or policies related to the strike. (e.g., “I am fully against this strike.)
If the comment falls under these categories, we say “Align with Company” is true.

The key indicator is that the author’s stance supports the company’s position rather than aligning with community dissent or collective protest. If the comment only expresses neutrality or critiques the company instead, we say “Align With Company” is false.


**None of the above**

If the comment does not fall under any of the above categories, we say “None of the above” is true.


### OUTPUT FORMAT

Represent your judgment as a single JSON object that matches this schema:

{
  "individual_voice": true/false,
  "collective_voice": true/false,
  "internal_deliberation": true/false,
  "align_with_company": true/false,
  "none_of_the_above": true/false,
  "reason": "Short explanation (1–2 sentences) for why you made this classification."
}

Do NOT explain your steps or thoughts. Your response must ONLY be the JSON object.

**Example 1**

Comment: "As a former moderator here, I'm saddened to see that the SE corporation still isn't really listening to the users who should be trusted the most.  The vast majority of the moderators I interacted with (or even watched work) are incredibly talented and intelligent people who care intensely about the community they curate."

Response: {"Individual Voice": false, "Collective Voice": true, "Internal Deliberation": false, "Align With Company": false,  "None of the above": false,

           "Reason": "The author speaks from the moderators’ and users’ perspective and expresses frustration toward the company for not listening to the community."}

**Example 2**

Comment: "As a former moderator here, I'm saddened to see that the SE corporation still isn't really listening to the users who should be trusted the most.  The vast majority of the moderators I interacted (or even watched work) are incredibly talented and intelligent people who care intensely about the community they curate."

Response: {"Individual Voice": false, "Collective Voice": true, "Internal Deliberation": false, "Align With Company": false,  "None of the above": false,

           "Reason": "The author speaks from the moderators’ and users’ perspective and expresses frustration toward the company for not listening to the community."}


**Example 3**

Comment: "I haven't read everything on this page. Maybe I'm missing something, but is there any evidence that any of this actually happened? As far as I can see, there isn't a link to even one AI - generated answer, let alone 10,000. What am I missing?"

Response: {"Individual Voice": false, "Collective Voice": false, "Internal Deliberation": true, "Align With Company": false,  "None of the above": false,

           "Reason": "This comment fits Internal Deliberation because the author is questioning evidence and trying to clarify factual claims without taking a stance for or against either side."}



**Example 4**

Comment: "From a longtime SO/SE user: I do not support this strike."

Response: {"Individual Voice": false, "Collective Voice": false, "Internal Deliberation": false, "Align With Company": true,  "None of the above": false,

           "Reason": "This comment directly expressed the author's disagreement with the community."}



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

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    # Build prompts
    prompts = []
    for c in comments:
        prompt = (
            SYSTEM_INSTRUCTIONS
            + "\n\nComment:\n"
            + c
            + "\n\nLabel:"
        )
        prompts.append(prompt)

    print("Sending prompts to LLM...")
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    labels = []
    for out in tqdm(outputs, desc="Collecting labels"):
        # vLLM output structure: one RequestOutput per prompt
        if not out.outputs:
            labels.append("ERROR:EMPTY_OUTPUT")
            continue
        text = out.outputs[0].text.strip()
        labels.append(text)

    df["voice_label"] = labels
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved labeled CSV to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
