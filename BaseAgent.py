import re
import yaml
from langchain_openai import ChatOpenAI
#from uilis import try_parse_json_object, try_parse_ast_to_json



class BaseAgent:
    def __init__(self):
        with open('/Users/feiou/llmopt/ev_agents/update_model/EV_agents/config.yaml', 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        
        self.llm = ChatOpenAI(
            temperature=0.1,
            model=self.config['Basic_Model_Config']['model'],
            openai_api_key=self.config['Basic_Model_Config']['API_KEY'],
            openai_api_base=self.config['Basic_Model_Config']['API_BASE_URL']
        )   
        

    


    def extract_json_fomrat(self, response):
        try:
        # Try to evaluate the response as a Python expression
            evaluated_response = eval(response)
        
        # Check if the evaluated response is a dictionary
            if isinstance(evaluated_response, dict):
                return evaluated_response
        except:
        # If evaluation fails or the response is not a dictionary, proceed with regex parsing
            pass
        
    # Define the regex pattern to match JSON enclosed in triple backticks
        PATTERN = re.compile(r"```(?:json\s+)?(\W.*?)```", re.DOTALL)
    
    # Search for the pattern in the response
        action_match = PATTERN.search(response)
    
        if action_match is not None:
        # Extract the matched JSON text and strip any leading/trailing whitespace
            json_object = action_match.group(1).strip()
            return eval(json_object)
        # Try to parse the JSON text into a Python object
        return None
    

    def extract_code(self, response):
    # Match code within ```python ... ``` or ``` ... ``` blocks
        pattern = r'```(?:python)?\s*(.*?)\s*```'
    
    # Find all matches in the input string
        code_blocks = re.findall(pattern, response, re.DOTALL)

        if len(code_blocks) == 0:
            # print(f'Parse code error! {input_string}')
            return response
        elif len(code_blocks) == 1:
            return code_blocks[0]

        code_blocks = [code for code in code_blocks if 'pip' not in code]
        return '\n'.join(code_blocks)
    
    def operate(self):
        pass