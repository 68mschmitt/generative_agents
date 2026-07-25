"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import os
import random
import openai
import time 

from utils import *

# Provider configuration is read from reverie/backend_server/utils.py.
# Defaults keep the original OpenAI behavior, while local Ollama can be enabled
# by setting llm_provider="ollama" and llm_api_base="http://localhost:11434/v1".
llm_provider = globals().get("llm_provider", "openai")
llm_api_base = globals().get("llm_api_base", None)
llm_chat_model = globals().get("llm_chat_model", "gpt-3.5-turbo")
llm_gpt4_model = globals().get("llm_gpt4_model", "gpt-4")
llm_completion_model = globals().get("llm_completion_model", None)
llm_embedding_model = globals().get("llm_embedding_model", "text-embedding-ada-002")
llm_trace_enabled = globals().get("llm_trace_enabled", False)
llm_trace_path = globals().get("llm_trace_path", "llm_trace.jsonl")
llm_embedding_cache_path = globals().get("llm_embedding_cache_path", None)

_prompt_template_cache = {}
_embedding_cache = {}
_embedding_persistent_cache_loaded = False
_embedding_persistent_cache_dirty = False

openai.api_key = globals().get("openai_api_key", "ollama")
if llm_api_base:
  openai.api_base = llm_api_base


def _chat_model(default_model):
  if llm_provider == "ollama":
    # Ollama does not have OpenAI model aliases such as gpt-3.5-turbo/gpt-4.
    return llm_gpt4_model if default_model == "gpt-4" else llm_chat_model
  return default_model


def _completion_model(default_model):
  if llm_provider == "ollama":
    # Route legacy text-davinci/text-curie prompts to the configured Ollama LLM.
    return llm_completion_model or llm_chat_model
  return default_model


def _embedding_model(default_model):
  if llm_provider == "ollama":
    return llm_embedding_model
  return default_model

def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def _text_len(value):
  try:
    if isinstance(value, (list, tuple)):
      return sum(_text_len(v) for v in value)
    return len(str(value))
  except Exception:
    return 0


def _write_llm_trace(provider, model, call_kind, input_length, elapsed, success, exception=None):
  if not llm_trace_enabled:
    return
  record = {
    "timestamp": time.time(),
    "provider": provider,
    "model": model,
    "call_kind": call_kind,
    "elapsed_seconds": elapsed,
    "input_length": input_length,
    "success": success,
  }
  if exception:
    record["exception"] = str(exception)
  try:
    trace_dir = os.path.dirname(llm_trace_path)
    if trace_dir:
      os.makedirs(trace_dir, exist_ok=True)
    with open(llm_trace_path, "a") as f:
      f.write(json.dumps(record) + "\n")
  except Exception as e:
    print("LLM TRACE WRITE FAILED", e)


def _trace_llm_call(model, call_kind, input_value, fn):
  start = time.time()
  try:
    result = fn()
    _write_llm_trace(llm_provider, model, call_kind, _text_len(input_value), time.time() - start, True)
    return result
  except Exception as e:
    _write_llm_trace(llm_provider, model, call_kind, _text_len(input_value), time.time() - start, False, e)
    raise


def ChatGPT_single_request(prompt): 
  temp_sleep()
  model = _chat_model("gpt-3.5-turbo")

  def _request():
    completion = openai.ChatCompletion.create(
      model=model, 
      messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
  return _trace_llm_call(model, "chat", prompt, _request)


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  model = _chat_model("gpt-4")
  try: 
    def _request():
      completion = openai.ChatCompletion.create(
      model=model, 
      messages=[{"role": "user", "content": prompt}]
      )
      return completion["choices"][0]["message"]["content"]
    return _trace_llm_call(model, "chat", prompt, _request)
  
  except Exception as e: 
    print ("ChatGPT ERROR", e)
    return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  model = _chat_model("gpt-3.5-turbo")
  try: 
    def _request():
      completion = openai.ChatCompletion.create(
      model=model, 
      messages=[{"role": "user", "content": prompt}]
      )
      return completion["choices"][0]["message"]["content"]
    return _trace_llm_call(model, "chat", prompt, _request)
  
  except Exception as e: 
    print ("ChatGPT ERROR", e)
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  model = _completion_model(gpt_parameter["engine"])
  try: 
    if llm_provider == "ollama":
      # Ollama's OpenAI-compatible server is much more reliable through the
      # chat endpoint than the legacy completions endpoint used by this 2023
      # codebase. Treat old text-davinci-style prompts as one user message.
      def _chat_request():
        completion = openai.ChatCompletion.create(
          model=model,
          messages=[{"role": "user", "content": prompt}],
          temperature=gpt_parameter["temperature"],
          max_tokens=gpt_parameter["max_tokens"],
          top_p=gpt_parameter["top_p"],
          frequency_penalty=gpt_parameter["frequency_penalty"],
          presence_penalty=gpt_parameter["presence_penalty"],
          stream=gpt_parameter["stream"],
          stop=gpt_parameter["stop"],)
        return completion["choices"][0]["message"]["content"]
      return _trace_llm_call(model, "chat", prompt, _chat_request)

    def _completion_request():
      response = openai.Completion.create(
                  model=model,
                  prompt=prompt,
                  temperature=gpt_parameter["temperature"],
                  max_tokens=gpt_parameter["max_tokens"],
                  top_p=gpt_parameter["top_p"],
                  frequency_penalty=gpt_parameter["frequency_penalty"],
                  presence_penalty=gpt_parameter["presence_penalty"],
                  stream=gpt_parameter["stream"],
                  stop=gpt_parameter["stop"],)
      return response.choices[0].text
    return _trace_llm_call(model, "completion", prompt, _completion_request)
  except Exception as e: 
    print ("LLM REQUEST FAILED", e)
    return "LLM REQUEST FAILED"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  if prompt_lib_file in _prompt_template_cache:
    prompt = _prompt_template_cache[prompt_lib_file]
  else:
    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    _prompt_template_cache[prompt_lib_file] = prompt
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def _extract_json_object(text):
  text = str(text or "").strip()
  start = text.find("{")
  end = text.rfind("}")
  if start == -1 or end == -1 or end <= start:
    raise ValueError("No JSON object found in LLM response")
  return json.loads(text[start:end+1])


def safe_json_request(prompt, schema_name=None, fail_safe=None, repeat=3, verbose=False):
  json_prompt = prompt
  if schema_name:
    json_prompt += f"\nReturn only valid JSON for schema: {schema_name}."
  for i in range(repeat):
    try:
      response = ChatGPT_request(json_prompt)
      if verbose:
        print(response)
      return _extract_json_object(response)
    except Exception as e:
      if verbose:
        print("JSON REQUEST FAILED", i, e)
  return fail_safe


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt): 
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def _load_persistent_embedding_cache():
  global _embedding_persistent_cache_loaded
  if _embedding_persistent_cache_loaded or not llm_embedding_cache_path:
    return
  _embedding_persistent_cache_loaded = True
  try:
    if os.path.exists(llm_embedding_cache_path):
      with open(llm_embedding_cache_path, "r") as f:
        data = json.load(f)
      for key, vector in data.items():
        provider, model, text = key.split("\t", 2)
        _embedding_cache[(provider, model, text)] = vector
  except Exception as e:
    print("LLM EMBEDDING CACHE LOAD FAILED", e)


def _save_persistent_embedding_cache():
  if not llm_embedding_cache_path:
    return
  try:
    cache_dir = os.path.dirname(llm_embedding_cache_path)
    if cache_dir:
      os.makedirs(cache_dir, exist_ok=True)
    data = {"\t".join(key): vector for key, vector in _embedding_cache.items()}
    with open(llm_embedding_cache_path, "w") as f:
      json.dump(data, f)
  except Exception as e:
    print("LLM EMBEDDING CACHE SAVE FAILED", e)


def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  normalized_text = " ".join(text.split())
  embedding_model = _embedding_model(model)
  cache_key = (llm_provider, embedding_model, normalized_text)
  _load_persistent_embedding_cache()
  if cache_key in _embedding_cache:
    return _embedding_cache[cache_key]

  def _request():
    return openai.Embedding.create(
            input=[normalized_text], model=embedding_model)['data'][0]['embedding']
  vector = _trace_llm_call(embedding_model, "embedding", normalized_text, _request)
  _embedding_cache[cache_key] = vector
  _save_persistent_embedding_cache()
  return vector


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















