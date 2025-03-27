from llama_stack import LlamaStackAsLibraryClient
from llama_stack.apis.sdg_functions import SDGFn

from llama_stack_client import LlamaStackClient

client = LlamaStackAsLibraryClient("remote-vllm")
client.initialize()

# client = LlamaStackClient(
#     base_url="http://localhost:8321"
# )


# Register our input dataset
emotion_dataset_id = "emotions"
client.datasets.register(
    dataset_id=emotion_dataset_id,
    purpose="eval/question-answer",
    source={
        "type": "rows",
        "rows": [
            {
                "prompt":"im feeling rather rotten so im not very ambitious right now",
                "label":0,
                "questions_and_answers":
                [
                    {"answer":"joy","question":"I can\u2019t believe how everything turned out\u2014it\u2019s like all the stars aligned just for me."},
                    {"answer":"sadness","question":"It\u2019s hard to keep going when every step feels heavier than the last, like I\u2019m walking against a strong tide."},
                    {"answer":"anger","question":"After all I did, this is how they repay me? It\u2019s frustrating beyond words."},
                    {"answer":"fear","question":"I\u2019m not sure what\u2019s going to happen next, but something about it just feels wrong and unsettling."},
                    {"answer":"surprise","question":"I really didn\u2019t see that coming\u2014it\u2019s like life threw a curveball out of nowhere."},
                    {"answer":"love","question":"There\u2019s just something about them; I can\u2019t quite explain it, but I feel this warmth every time they\u2019re around."}
                ],
                "task_description":"Your task is to analyze sentences and classify them into one of six primary emotion categories: \u201cjoy,\u201d \u201csadness,\u201d \u201canger,\u201d \u201cfear,\u201d \u201csurprise,\u201d or \u201clove.\u201d Each sentence represents a distinct emotional expression, and you should interpret contextual clues to identify the underlying emotion accurately.\n\n1. joy: Sentences expressing happiness, contentment, or a sense of fulfillment. Look for words or phrases that indicate pleasure or satisfaction.\n2. sadness: Sentences indicating a sense of loss, disappointment, or sorrow. Expressions may include words that convey emotional pain or melancholy.\n3. anger: Sentences with frustration, irritation, or outrage. Often includes language signaling dissatisfaction or indignation.\n4. fear: Sentences showing anxiety, worry, or apprehension. Common indicators are words related to uncertainty or perceived threat.\n5. surprise: Sentences expressing shock, disbelief, or unexpectedness. Often marked by phrases indicating astonishment or being taken off guard.\n6. love: Sentences conveying affection, admiration, or warmth toward a person, place, or concept. Words often reflect strong positive emotions or attachment.\n\nPay close attention to subtle language that may suggest underlying emotions, such as metaphors or tone. Consider context clues that give depth to the emotional sentiment. In cases where the emotion may seem ambiguous, prioritize the strongest linguistic indicators in the text, focusing on implied tone and context to select the closest matching emotion.",
                "simple_task_description":"Classify sentences into one of five categories: joy, sadness, anger, fear, surprise, love.","ground_truth":"sadness"
            },
            {
                "prompt":"im updating my blog because i feel shitty","label":0,
                "questions_and_answers":
                [
                    {"answer":"joy","question":"I can\u2019t believe how everything turned out\u2014it\u2019s like all the stars aligned just for me."},
                    {"answer":"sadness","question":"It\u2019s hard to keep going when every step feels heavier than the last, like I\u2019m walking against a strong tide."},
                    {"answer":"anger","question":"After all I did, this is how they repay me? It\u2019s frustrating beyond words."},
                    {"answer":"fear","question":"I\u2019m not sure what\u2019s going to happen next, but something about it just feels wrong and unsettling."},
                    {"answer":"surprise","question":"I really didn\u2019t see that coming\u2014it\u2019s like life threw a curveball out of nowhere."},
                    {"answer":"love","question":"There\u2019s just something about them; I can\u2019t quite explain it, but I feel this warmth every time they\u2019re around."}
                ],
                "task_description":"Your task is to analyze sentences and classify them into one of six primary emotion categories: \u201cjoy,\u201d \u201csadness,\u201d \u201canger,\u201d \u201cfear,\u201d \u201csurprise,\u201d or \u201clove.\u201d Each sentence represents a distinct emotional expression, and you should interpret contextual clues to identify the underlying emotion accurately.\n\n1. joy: Sentences expressing happiness, contentment, or a sense of fulfillment. Look for words or phrases that indicate pleasure or satisfaction.\n2. sadness: Sentences indicating a sense of loss, disappointment, or sorrow. Expressions may include words that convey emotional pain or melancholy.\n3. anger: Sentences with frustration, irritation, or outrage. Often includes language signaling dissatisfaction or indignation.\n4. fear: Sentences showing anxiety, worry, or apprehension. Common indicators are words related to uncertainty or perceived threat.\n5. surprise: Sentences expressing shock, disbelief, or unexpectedness. Often marked by phrases indicating astonishment or being taken off guard.\n6. love: Sentences conveying affection, admiration, or warmth toward a person, place, or concept. Words often reflect strong positive emotions or attachment.\n\nPay close attention to subtle language that may suggest underlying emotions, such as metaphors or tone. Consider context clues that give depth to the emotional sentiment. In cases where the emotion may seem ambiguous, prioritize the strongest linguistic indicators in the text, focusing on implied tone and context to select the closest matching emotion.",
                "simple_task_description":"Classify sentences into one of five categories: joy, sadness, anger, fear, surprise, love.","ground_truth":"sadness"
             },
        ],
    },
)

# Register a custom SDG Function
pipeline_yaml = """
- block_type: LLMBlock
  block_config:
    block_name: gen_responses
    config_path: detailed_description_icl.yaml
    model_id: meta-llama/Llama-3.2-3B-Instruct
    output_cols:
      - output
  gen_kwargs:
    max_tokens: 500
    temperature: 0
    extra_body:
      guided_choice:
        - "joy"
        - "sadness"
        - "anger"
        - "fear"
        - "love"
  drop_duplicates:
    - prompt
"""
detailed_description_icl_yaml = """
system: ~
introduction: |
  Task Description: {{ task_description }}
principles: ~
examples: |
  To better assist you with this task, here are some examples:
  {% if questions_and_answers is defined %}
  {% for sample in questions_and_answers %}
  [Start of Question]
  {{ sample.question }}
  [End of Question]

  [Start of Output]
  {{ sample.answer }}
  [End of Output]
  {% endfor %}
  {% else %}
  [Start of Question]
  {{ seed_question }}
  [End of Question]

  [Start of Output]
  {{ seed_response }}
  [End of Output]
  {% endif %}
generation: |
  Here is the query for annotation:
  [Start of Question]
  {{ prompt }}
  [End of Question]
start_tags: [""]
end_tags: [""]
"""

meta_llama_chat_template = """{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"""

annotation_sdg_fn_id = "annotation_emotion_example"
client.sdg_functions.register(
    sdg_fn_id=annotation_sdg_fn_id,
    description="An example annotation SDG Function",
    provider_id="instructlab-sdg",
    params={
        "type": "instructlab_sdg",
        "pipeline_yaml": pipeline_yaml,
        "extra_configs": {
            "detailed_description_icl.yaml": detailed_description_icl_yaml,
        },
        "chat_templates": {
            "meta-llama": meta_llama_chat_template,
        },
    },
)

response = client.synthetic_data_generation.generate(
    dataset_id=emotion_dataset_id,
    sdg_fn_id=annotation_sdg_fn_id,
)

print("\n")
for sample in response.synthetic_data:
    print(f'Prompt: {sample["prompt"]}')
    print(f'Ground truth: {sample["ground_truth"]}')
    print(f'Output: {sample["output"]}')
    print("\n")

client.sdg_functions.unregister(
    sdg_fn_id=annotation_sdg_fn_id
)
