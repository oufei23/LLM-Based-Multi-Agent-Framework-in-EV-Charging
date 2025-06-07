from BaseAgent import BaseAgent
from langchain_core.prompts import PromptTemplate

class ModelingEvaluator(BaseAgent):
    def __init__(self,user_parameters,modeling_parameters):
        super().__init__()
        self.user_parameters = user_parameters
        self.modeling_parameters = modeling_parameters

        self.modeling_check_template = PromptTemplate.from_template(self.config['prompt_templates']['modeling_check_template'])
        self.modeling_refine_template = PromptTemplate.from_template(self.config['prompt_templates']['modeling_refine_template'])

        self.modeling_para_schema = PromptTemplate.from_template(self.config['json_schemas']['modeling_para_schema'])
        self.modeling_check_schema = PromptTemplate.from_template(self.config['json_schemas']['modeling_check_schema'])
        

    def get_OP_descrpition(self):
        # Get the description of the optimization problem, either fast charging mode or slow charging mode
        if self.user_parameters['快慢充偏好'] == True:
            return self.config['math_problem']['problem_1']
        return self.config['math_problem']['problem_2']
    
    def check_modeling_correctness(self,modeling_parameters):
        chain = self.modeling_check_template | self.llm
        response = chain.invoke({"modeling_parameters":modeling_parameters, "OP_descrpition": self.get_OP_descrpition(),"user_para": self.user_parameters,"json_shema": self.modeling_check_schema})
        #print(response.content)
        results = self.extract_json_fomrat(response.content)
        return results
    
        
    def refine_modeling(self,init_modeling_parameter,advices):
        chain = self.modeling_refine_template | self.llm
        response = chain.invoke({"modeling_parameters" : init_modeling_parameter, "OP_descrpition": self.get_OP_descrpition(),"advices": advices, "user_para": self.user_parameters, "json_shema": self.modeling_para_schema})
        refined_exp_parameter = self.extract_json_fomrat(response.content)
        return refined_exp_parameter
    

    def operate(self):
        attempt = 0
        max_attempts = 10
        modeling_parameters = self.modeling_parameters
        check_history = []
        while attempt < max_attempts:
            check_results = self.check_modeling_correctness(modeling_parameters)

            check_history.append(check_results)
    
            m_valid,advices = check_results['PredictionResult'],check_results['comment']

            if m_valid == 1:
                #print("Modeling extraction is correct.")
                break
            else:
                
                if m_valid == 0:
                    print(f"Attempt {attempt + 1}: Expression is not valid. Refining...")
                    modeling_parameters = self.refine_modeling(modeling_parameters, advices)

            attempt += 1
            if attempt == max_attempts:
                #print("Maximum number of attempts reached. Stopping refinement process.")
                break
        return modeling_parameters,check_history


        
        
    
if __name__ == "__main__":
    
    
    user_parameters = {'持续时长': 5, '目标电量': 90, '期望续航': 500, '快慢充偏好': True, '用户预算': 100, '品牌型号': '小米su7max', '电池容量': 66, '续航里程': 800, '快充支持': True, '慢充支持': True, '快充功率': 13}
    model_parameters = {'objective_function': 'minimize c P', 'decision_variable': 't^f_i, x_f', 'constraint_expression': {'constraint_1': '\\sum_{i=1}^{T} t^{f}_i \\leq T', 'constraint_2': 'P = \\sum_{i=1}^{T} x_{f} t^{f}_{i} \\geq P_{req}', 'constraint_3': 'C \\leq C_b'}, 'constant_value': 'c = [0.75, 0.8, 0.78, 0.82, 0.85], T = 5, P_{req} = 66.00999999999999, C_b = 80', 'variable_explanation': 'c: real time changing electricity price, t^f_i: time for fast charging at time slot i, x_f: fast charging power, P: total energy charged, P_{req}: required energy for charging, C: total cost, C_b: budget of the user'}
    agent_m  = ModelingEvaluator (user_parameters=user_parameters,modeling_parameters=model_parameters)
    final_result = agent_m.operate()
    print(final_result[0])
    print('result:',final_result[1])

