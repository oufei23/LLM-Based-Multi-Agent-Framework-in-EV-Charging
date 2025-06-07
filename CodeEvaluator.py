from BaseAgent import BaseAgent
from langchain_core.prompts import PromptTemplate

class CodeEvaluator(BaseAgent):
    def __init__(self,generated_code,modeling_parameters):
        super().__init__()
        self.generated_code = generated_code
        self.modeling_parameters = modeling_parameters
        self.code_check_template = PromptTemplate.from_template(self.config['prompt_templates']['code_check_template'])
        self.code_refine_template = PromptTemplate.from_template(self.config['prompt_templates']['code_refine_template'])
        self.modeling_check_schema = PromptTemplate.from_template(self.config['json_schemas']['modeling_check_schema'])

    # def provide_comment(self):
    #     chain = self.eval_template | self.llm
    #     response = chain.invoke({"code_block" : self.generated_code})
    #     return response.content
    
    def check_code_correctness(self,code):
        chain = self.code_check_template | self.llm
        response = chain.invoke({"modeling_parameters":self.modeling_parameters, "code_block": code, "json_shema": self.modeling_check_schema})
        #print(response.content)
        results = self.extract_json_fomrat(response.content)
        return results
    
    def refine_code(self,init_code,advices):
        chain = self.code_refine_template | self.llm
        response = chain.invoke({"code_block" : init_code, "advices": advices,"modeling_parameter":self.modeling_parameters})
        return self.extract_code(response.content)
    
    def operate(self):
        attempt = 0
        max_attempts = 10
        code = self.generated_code
        check_history = []
        #modeling_parameter = self.modeling_parameters
        while attempt < max_attempts:
            check_results = self.check_code_correctness(code)

            #print(check_results)
            check_history.append(check_results)

            c_valid, advices = check_results['PredictionResult'],check_results['comment']

            if c_valid == 1:
                #print("Coding generation is correct.")
                break
            else:
                
                if c_valid == 0:
                    #print(f"Attempt {attempt + 1}: Code is not valid. Refining...")
                    code = self.refine_code(code,advices)

            attempt += 1
            if attempt == max_attempts:
                #print("Maximum number of attempts reached. Stopping refinement process.")
                pass
        return code,check_history
    
if __name__ == "__main__":
    generated_code = """
    import gurobipy as gp
from gurobipy import GRB

# 1. 建立新模型
m = gp.Model(name="ChargingOptimization")

# 2. 建立决策变量
# 因为$t^{2}_i$在0和1之间，并且是二进制变量，所以我们使用GRB.BINARY
t_2 = m.addMVar(shape=8, vtype=GRB.BINARY, name="t_2")

# 3. 设立优化目标函数
# 根据参数，我们的目标是最小化 c P，其中 c 是实时电价，P 是总充电量
c = [0.75, 0.8, 0.78, 0.82, 0.85, 0.9, 0.75, 0.8]  # 实时电价
x_s = 7  # 慢充功率
P = sum(c[i] * t_2[i] for i in range(8))  # 定义P的表达式
m.setObjective(P, GRB.MINIMIZE)

# 4. 创建约束条件
# 约束1: \(\sum_{i=1}^{8} t^{2}_i \leq 8\)
m.addConstr(sum(t_2[i] for i in range(8)) <= 8, name="constraint_1")

# 约束2: \(P = \sum_{i=1}^{8} x_{s} t^{2}_{i} \geq 90\)
m.addConstr(P >= 90, name="constraint_2")

# 约束3: \(C \leq 100\)
# 这里C是电费，我们需要计算总电费不超过预算
C = sum(c[i] * x_s * t_2[i] for i in range(8))
m.addConstr(C <= 100, name="constraint_3")

# 5. 优化模型
m.optimize()

# 6. 输出优化结果
if m.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for v in m.getVars():
        print(f"{v.varName}: {v.x}")
    print(f"Objective: {m.ObjVal}")
else:
    print("No optimal solution found.")    
    """
    modeling_parameter = {'objective_function': 'minimize c P', 'decision_variable': 't_2', 'constraint_expression': {'constraint_1': '$\\sum_{i=1}^{8} t^{2}_i \\leq 8$', 'constraint_2': 'P = \\sum_{i=1}^{8} x_{s} t^{2}_{i} \\geq 90', 'constraint_3': 'C \\leq 100'}, 'constant_value': 'real time electricity price c = [0.75, 0.8, 0.78, 0.82, 0.85, 0.9, 0.75, 0.8], charging duration T = 8, slow charging power x_s = 7, required energy for charging P_{req} = 90, user budget C_b = 100', 'variable_explanation': 'where $c$ is real time changing electricity price, and dim(c) = T, $t_2$ is time for slow charging $t^{2}_{i}$ = [0,1], $0\\leq i \\leq T$, and $T = dim(t^2)$, $x_s$ is slow charging power, $P_{req}$ is required energy for charging, and $C_b$ is the budget of the user'}
    agent_ce  = CodeEvaluator (generated_code,modeling_parameter)
    code = agent_ce.operate()
    print(code)
