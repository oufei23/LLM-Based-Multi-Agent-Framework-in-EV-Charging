import yaml
from BaseAgent import BaseAgent
from langchain_core.prompts import PromptTemplate

class MathEvaluator(BaseAgent):
    def __init__(self,user_parameter,modeling_parameters):
        super().__init__()
        self.user_parameter = user_parameter
        self.modeling_parameters = modeling_parameters
        self.math_check_template = PromptTemplate.from_template(self.config['prompt_templates']['math_check_template'])
        
        self.modeling_exp_shema = PromptTemplate.from_template(self.config['json_schemas']['modeling_exp_schema'])

        # Load configuration from YAML file
        with open('/Users/feiou/llmopt/ev_agents/update_model/EV_agents/config.yaml', 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

    def get_OP_descrpition(self):
        # Get the description of the optimization problem, either fast charging mode or slow charging mode
        if self.user_parameters['快慢充偏好'] == True:
            return self.config['math_problem']['problem1']
    
    def check_correctness(self):
        chain = self.math_check_template | self.llm
        response = chain.invoke({"modeling_parameter" : self.modeling_parameters, "OP_descrpition": self.get_OP_descrpition, "json_shema": self.modeling_exp_shema})
        comment = response.content
        print(comment)
        try:
            modeling_expression = self.extract_json_fomrat(response.content)
            return self.extract_json_fomrat(modeling_expression)
        except:
            return comment
        
    
    def output_para(self):

        return None
    
    def operate(self):
        modeling_expression = self.check_correctness()
        return modeling_expression
    
if __name__ == "__main__":
    
    
    user_parameter = {'持续时长': 8, '目标电量': 90, '期望续航': 500, '快慢充偏好': False, '用户预算': 100, '品牌型号': '小米su7max', '电池容量': 101, '续航里程': 800, '快充支持': True, '慢充支持': True}
    modeling_parameters = { 'objective_function': 'c^t * P', 'decision_variable': 't^1', 'constraints': {'constraint_1': '$\\sum_{i=1}^{T} t^{1}_i \\leq T$', 'constraint_2': '$\\sum_{i=1}^{T} x_{f} t^{1}_i \\geq P$', 'constraint_3': 'C \\leq C_budget'}, 'variable_explanation': {'c': 'real time changing electricity price vector, dim(c) = T', 't^1': 'charging time decision variable vector, t^{1}_i \\in [0,1], 0\\leq i \\leq T', 'T': 'total number of time slots', 'x_f': 'fast charging power', 'P': 'required energy for charging', 'C': 'total cost, C = c^t * P', 'C_budget': 'budget of the user'}}
    agent_m  = MathEvaluator (user_parameter=user_parameter,modeling_parameters=modeling_parameters)
    agent_m.operate()
