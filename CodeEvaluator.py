from BaseAgent import BaseAgent
from langchain_core.prompts import PromptTemplate

class CodeEvaluator(BaseAgent):
    def __init__(self,generated_code):
        super().__init__()
        self.generated_code = generated_code
        self.eval_template = PromptTemplate.from_template(self.config['prompt_templates']['evaluate_template'])
        self.refine_template = PromptTemplate.from_template(self.config['prompt_templates']['refine_template'])

    def provide_comment(self):
        chain = self.eval_template | self.llm
        response = chain.invoke({"code_block" : self.generated_code})
        return response.content
    
    def refine_code(self):
        chain = self.refine_template | self.llm
        response = chain.invoke({"code_block" : self.generated_code, "comment": self.provide_comment()})
        return self.extract_code(response.content)
    
    def operate(self):
        print("原始代码块: ", self.generated_code)
        comment = self.provide_comment()
        print("评价: ", comment)
        refined_code = self.refine_code()
        print("优化后代码块: ", refined_code)
        return refined_code
    
if __name__ == "__main__":
    generated_code = """
        import gurobipy as gp
        from gurobipy import GRB
        import numpy as np

        # 参数数值
        T = 8
        P = 90.9
        C_budget = 100
        c = np.array([0.75, 0.8, 0.78, 0.82, 0.85, 0.9, 0.88, 0.86])
        x_f = 1  # 假设x_f为固定值1，因为它是充电功率，可以简化处理

        # 1. 建立新模型
        m = gp.Model(name="ChargingOptimization")

        # 2. 建立决策变量
        t = m.addMVar(shape=T, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="t")

        # 3. 设立优化目标函数
        # 根据表达式 'c^t * P'，但在这里我们使用 'c * t' 因为P是常量，在最后计算总成本时乘以P
        obj = c * t
        m.setObjective(obj @ x_f, GRB.MINIMIZE)  # 最小化总成本

        # 4. 创建约束条件
        # 约束1: 总充电时间不超过T
        m.addConstr(t.sum() <= T, name="constraint_1")

        # 约束2: 充电总能量至少为P
        m.addConstr(x_f * t.sum() >= P, name="constraint_2")

        # 约束3: 总成本不超过预算
        # 由于目标函数是 c * t，我们需要确保 c * t * x_f <= C_budget
        # 这里我们直接在目标函数中考虑了x_f，所以这里不需要额外的约束

        # 5. 优化模型
        m.optimize()

        # 6. 输出优化结果
        if m.status == GRB.OPTIMAL:
            print(f"Total Cost: {m.ObjVal}")
            print("Charging times: ")
            for v in t:
                print(f"{v.varName} = {v.x}")
        else:
            print("Optimization was stopped with status:", m.status)
            agent_ce = CodeEvaluator()"""
    agent_ce  = CodeEvaluator (generated_code)
    agent_ce.operate()
