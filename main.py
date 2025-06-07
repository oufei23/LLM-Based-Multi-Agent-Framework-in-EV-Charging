import subprocess
from InfoCollector import InfoCollector
from ModelingExtractor import ModelingExtractor
from MathEvaluator import MathEvaluator
from CodeGenerator import CodeGenerator
from CodeEvaluator import CodeEvaluator

def query_to_code(input_query):

    #information collection stage
    Agent_I = InfoCollector(input_query=input_query)
    user_para, ev_para = Agent_I.operate()
    print(type(user_para), type(ev_para))
    #if type(user_para) == str:
        #user_para = eval(user_para)
    #if type(ev_para) == str:
        #ev_para = eval(ev_para)

    parameters = {**user_para, **ev_para}

    #modeling parameter extraction stage
    Agent_M_a = ModelingExtractor(input_parameters=parameters)
    modeling_exp,modeling_value = Agent_M_a.operate()
    print("提取到的建模表达式为: ", modeling_exp)
    print("提取到的建模相关参数数值为: ", modeling_value)

    Agent_M_e = MathEvaluator(user_parameter=user_para,modeling_parameters=modeling_exp)
    updated_modeling_para = Agent_M_e.operate()
    print("更新后的建模参数为: ", updated_modeling_para)

    #code generation stage
    Agent_C_a = CodeGenerator(modeling_exp=updated_modeling_para,modeling_para_value=modeling_value)
    generated_code = Agent_C_a.operate()
    Agent_C_e = CodeEvaluator(generated_code)
    final_code = Agent_C_e.operate()
    return final_code

def code_to_sol(code):
    with open('code.py', 'w') as file:
        file.write(code)
    p = subprocess.Popen('python3 code.py',
                     shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,
                     encoding='utf-8'
                     )

    # 输出stdout
    print(p.communicate()[0])


if __name__ == "__main__":
    input_query = "我想让我的车xiaomisu7max充电大约充8小时,我的预算不多，大概100元"
    code = query_to_code(input_query)
    sol = code_to_sol(code)
    print(sol)





