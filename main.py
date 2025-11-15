import os
import json
from openai import OpenAI
from tools import function_register
from tools.registerData import registerData
from tools.dataLoad import dataLoad, load_MDD_edf, load_Sleep_edf
from tools.baseInfo import baseInfo, get_age_factor
from prompt import getSystemPrompt
from utils.parseCalling import extract_tool_calls, has_config_parameter
from utils.messageMerge import messageMerge
import yaml
import time

class EEGAgent:
    def __init__(self, config_path: str, file_name: str, api_key: str, base_url: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        with open(os.path.join(self.config['report_template_path'], 'report_template.yaml'), 'r') as file:
            report_template = yaml.safe_load(file)
        self.file_path = os.path.join(self.config['dataPath'], file_name)

        ### Load Data... ###
        # data = load_MDD_edf(self.file_path, self.config)
        data = dataLoad(self.file_path, self.config)
        # data = load_Sleep_edf(self.file_path, self.config)
        registerData(data)

        ### generate baseInfo ###
        self.info = baseInfo(self.file_path)
        self.tool_schemas = function_register.export_tool_schemas()

        ### init ###
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        ### construct system prompt ###
        prior_knowlwdge = self.config['prior knowledge'].copy()
        if 'age' in self.info.keys():
            age_factor = get_age_factor(self.info['age'], self.config['prior knowledge']['Age factor'])
            prior_knowlwdge['Age factor'] = age_factor
        self.system_prompt = getSystemPrompt(self.info, self.tool_schemas, prior_knowlwdge, report_template)
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
    
    def prepare_user_message(self, user_query: str):
        self.messages.append({'role': 'user', 'content': user_query})

    def call_model(self, model="qwen-plus-2025-09-11"): # qwen3-14b qwen3-32b qwen3-235b-a22b qwen-plus-2025-07-28
        completion = self.client.chat.completions.create(
            model=model,
            messages=self.messages,
            # extra_body={"enable_thinking": False},
            timeout=60
        )
        response = completion.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": response
        })
        return response

    def handle_tool_calls(self, response):
        calls = extract_tool_calls(response)
        if not calls:
            return False 

        function_return = []
        for call in calls:
            function_name = call['name'].strip()
            args = call['args'].copy()
            function = function_register.get_function(function_name)
            if not function:
                print(f"No function named {function_name}")
                continue

            ### automated import config parameters ###
            if has_config_parameter(function) and 'config' not in args:
                args['config'] = self.config

            try:
                output = function(**args)
                new_call = call.copy()
                new_call['return'] = output
                function_return.append(new_call)
            except Exception as e:
                print(f"{function_name} execution unsuccessful: {e}")
                raise e
        messageMerge(function_return, self.messages)
        return True  

    def run(self, user_query):
        from RAG.embedder import BGEEmbedder
        from RAG.searcher import FaissSearcher
        
        # RAG
        embedder = BGEEmbedder()
        searcher = FaissSearcher("RAG/faiss.index", "RAG/chunks.pkl")
        query_vector = embedder.encode([user_query])[0]
        threshold = self.config.get("similarity_threshold")
        results = searcher.search(query_vector, top_k=3)
        for rank, (text, score) in enumerate(results, 1):
            if score >= threshold:
                self.messages[0]['content'] += f"{rank}. {text}\n"
        
        self.prepare_user_message(user_query)

        total_rounds = 0
        total_time = 0.0
        local_tool_time = 0.0

        start_total = time.time()
        while True:
            round_start = time.time()
            response = self.call_model()
            round_end = time.time()
            total_rounds += 1
            total_time += (round_end - round_start)

            tool_start = time.time()
            has_more_tools = self.handle_tool_calls(response)
            local_tool_time += (time.time() - tool_start)

            if not has_more_tools:
                break

        total_duration = time.time() - start_total

        return {
            "response": response,
            "rounds": total_rounds,
            "model_time": total_time,
            "local_tool_time": local_tool_time,
            "total_time": total_duration
        }
        

if __name__ == "__main__":
    agent = EEGAgent(
        config_path="config/config.json",
        file_name="spsw_077_a_1.edf",
        api_key = "***",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    user_question = "Can epileptic discharges be observed within the first minute? If so, where?"
    print("Human:", user_question)
    agent.run(user_question)
