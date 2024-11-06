import numpy as np
from BaseAgent import BaseAgent
from langchain_core.prompts import PromptTemplate




class ModelingExtractor(BaseAgent):
    def __init__(self, input_parameters):
        super().__init__()
        self.input_parameters = input_parameters
        
        # Load configuration from YAML file
        

        self.modeling_exp_template = PromptTemplate.from_template(self.config['prompt_templates']['modeling_exp_template'])
        self.modeling_value_template = PromptTemplate.from_template(self.config['prompt_templates']['modeling_value_template'])

        self.modeling_exp_shema = self.config['json_schemas']['modeling_exp_schema']
        self.modeling_value_shema = self.config['json_schemas']['modeling_value_schema']

    def get_OP_descrpition(self):
        # Get the description of the optimization problem, either fast charging mode or slow charging mode
        if self.input_parameters['快慢充偏好'] == True:
            return self.config['math_problem']['problem1']
        
        return self.config['math_problem']['problem2']
    
    def get_realtime_eprice(self):
        # Get the real-time electricity price，randomly setted here in 24-hour time slot for experiment.
        price = np.array([0.75, 0.8, 0.78, 0.82, 0.85, 0.9, 0.88, 0.86, 
            0.9, 0.92, 0.95, 0.98, 1.0, 1.05, 1.02, 1.0, 
            0.97, 0.95, 0.93, 0.9, 0.87, 0.85, 0.8, 0.78])
        duration = self.input_parameters['持续时长']
        duration_price = price[:duration]
        return duration_price
    
    def get_modeling_expression(self):
        chain = self.modeling_exp_template | self.llm
        response = chain.invoke({"OP_descrpition" : self.get_OP_descrpition, "json_shema": self.modeling_exp_shema})
        modeling_expression = self.extract_json_fomrat(response.content)
        return modeling_expression
    
    def get_modeling_parameters(self,modeling_ex_description):
        chain = self.modeling_value_template | self.llm
        response = chain.invoke({"modeling_ex_descrpition" : modeling_ex_description,"parameters": self.input_parameters, "json_shema": self.modeling_value_shema})
        return self.extract_json_fomrat(response.content)



    def get_required_energy(self):
        required_battery_percent = (int(self.input_parameters['目标电量']))/100
        battery_capacity = float(self.input_parameters['电池容量'])
        # assuming the battery is 0% at the beginning
        required_energy = required_battery_percent * battery_capacity
        # 更新所需电量
        self.input_parameters['所需电量'] = required_energy
        return required_energy
    
    def operate(self):
        self.get_required_energy()
        modeling_ex_description = self.get_modeling_expression()
        print(modeling_ex_description)
        modeling_parameters = self.get_modeling_parameters(modeling_ex_description)
        print(modeling_parameters)
        modeling_parameters['实时电价'] = self.get_realtime_eprice()




if __name__ == "__main__":
    
    
    input_parameter = {'持续时长': 8, '目标电量': 90, '期望续航': 500, '快慢充偏好': False, '用户预算': 100, '品牌型号': '小米su7max', '电池容量': 101, '续航里程': 800, '快充支持': True, '慢充支持': True}
    agent_m  = ModelingExtractor (input_parameter)
    agent_m.operate()
    