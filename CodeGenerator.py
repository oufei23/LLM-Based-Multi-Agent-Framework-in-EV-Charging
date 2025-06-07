import numpy as np
from BaseAgent import BaseAgent
from langchain_core.prompts import PromptTemplate

class CodeGenerator(BaseAgent):
    def __init__(self,modeling_parameters):
        super().__init__()
        self.modeling_parameters = modeling_parameters
        self.code_template = PromptTemplate.from_template(self.config['prompt_templates']['code_template'])
    
    def generate_code(self):
        chain = self.code_template | self.llm
        response = chain.invoke({"modeling_parameters" : self.modeling_parameters})
        return self.extract_code(response.content)
    
    def operate(self):
        code = self.generate_code()
        #print(code)
        return code
    
if __name__ == "__main__":

    modeling_para = {'objective_function': 'minimize c P', 'decision_variable': 't_2', 'constraint_expression': {'constraint_1': '$\\sum_{i=1}^{8} t^{2}_i \\leq 8$', 'constraint_2': 'P = \\sum_{i=1}^{8} x_{s} t^{2}_{i} \\geq 90', 'constraint_3': 'C \\leq 100'}, 'constant_value': 'real time electricity price c = [0.75, 0.8, 0.78, 0.82, 0.85, 0.9, 0.75, 0.8], charging duration T = 8, slow charging power x_s = 7, required energy for charging P_{req} = 90, user budget C_b = 100', 'variable_explanation': 'where $c$ is real time changing electricity price, and dim(c) = T, $t_2$ is time for slow charging $t^{2}_{i}$ = [0,1], $0\\leq i \\leq T$, and $T = dim(t^2)$, $x_s$ is slow charging power, $P_{req}$ is required energy for charging, and $C_b$ is the budget of the user'}
    agent_c = CodeGenerator(modeling_para)
    agent_c.operate()