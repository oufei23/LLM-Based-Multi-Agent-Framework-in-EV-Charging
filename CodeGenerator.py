import numpy as np
from BaseAgent import BaseAgent
from langchain_core.prompts import PromptTemplate

class CodeGenerator(BaseAgent):
    def __init__(self,modeling_exp,modeling_para_value):
        super().__init__()
        self.modeling_exp = modeling_exp
        self.modeling_para_value = modeling_para_value
        self.code_template = PromptTemplate.from_template(self.config['prompt_templates']['code_template'])
    
    def generate_code(self):
        chain = self.code_template | self.llm
        response = chain.invoke({"modeling_exp" : self.modeling_exp,"modeling_para_value":self.modeling_para_value})
        return self.extract_code(response.content)
    
    def operate(self):
        code = self.generate_code()
        print(code)
        return code
    
if __name__ == "__main__":
    modeling_exp = { 'objective_function': 'c^t * P', 'decision_variable': 't^1', 'constraints': {'constraint_1': '$\\sum_{i=1}^{T} t^{1}_i \\leq T$', 'constraint_2': '$\\sum_{i=1}^{T} x_{f} t^{1}_i \\geq P$', 'constraint_3': 'C \\leq C_budget'}, 'variable_explanation': {'c': 'real time changing electricity price vector, dim(c) = T', 't^1': 'charging time decision variable vector, t^{1}_i \\in [0,1], 0\\leq i \\leq T', 'T': 'total number of time slots', 'x_f': 'fast charging power', 'P': 'required energy for charging', 'C': 'total cost, C = c^t * P', 'C_budget': 'budget of the user'}}
    modeling_para_value = {'持续时间(T)': 8, '所需能量(P)': 90.9, '用户预算(C_b)': 100, '实时电价': np.array([0.75, 0.8, 0.78, 0.82, 0.85, 0.9, 0.88, 0.86])}
    agent_c  = CodeGenerator (modeling_exp,modeling_para_value)
    agent_c.operate()