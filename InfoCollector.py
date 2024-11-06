
import yaml
from BaseAgent import BaseAgent
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma



class InfoCollector(BaseAgent):
    def __init__(self,input_query):
        super().__init__()
        self.input_query = input_query
        self.feedback = ""
        # Load configuration from YAML file
        with open('/Users/feiou/llmopt/ev_agents/update_model/EV_agents/config.yaml', 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Extract prompt templates and JSON schemas
        self.prompt_template1 = PromptTemplate.from_template(self.config['prompt_templates']['user_template'])
        self.prompt_template2 = PromptTemplate.from_template(self.config['prompt_templates']['ev_template'])
        self.json_shema1 = self.config['json_schemas']['user_schema']
        self.json_shema2 = self.config['json_schemas']['ev_schema']

        
    def get_user_para(self):

        #get user-related parameters

        chain = self.prompt_template1 | self.llm
        response = chain.invoke({"query" : self.input_query, "feedback": self.feedback, "json_shema": self.json_shema1})
        #print(response.content)
        return self.extract_json_fomrat(response.content)
    
    def get_ev_para(self):
        ev_context = self.extract_closest_data_piece(self.input_query)
        chain = self.prompt_template2 | self.llm
        response = chain.invoke({"query" : self.input_query, "context": ev_context, "json_shema": self.json_shema2})
        #print(response.content)
        return self.extract_json_fomrat(response.content)
    
    
    def get_feedback(self, para_list):
        
        feedback = ""

        # 定义问题模板
        questions = {
            '持续时长': "您似乎没有提及充电时长，请问您的预期充电持续时长是多久？: ",
            '目标电量': "您似乎没有提及目标电量，请问您有目标电量百分比相关的要求吗？: ",
            '期望续航': "您似乎没有提及期望续航，请问您是否对充完电后的预期续航有要求呢？: ",
            '快慢充偏好': "您似乎没有提及快慢充偏好，请问您是更加偏向快充还是慢充呢？: ",
            '用户预算': "您似乎没有提及用户预算，请问您对预算方面有要求吗: "
        }
        
        # 遍历para_list中的键值对
        for key, value in para_list.items():
            
            if (value == 0 or not isinstance(value, (float, int))) and key != '快慢充偏好':
                question = questions[key]
                user_input = input(question)
                feedback += f"{key}: {user_input}\n"
            elif key == '快慢充偏好' and value not in [True, False]:
                question = questions[key]
                user_input = input(question)
                feedback += f"{key}: {user_input}\n"

        return feedback.strip()

    def operate(self):
        init_response = self.get_user_para()
        print("提取到的初始用户参数: ", init_response)
        self.feedback = self.get_feedback(init_response)
        final_user_parameters = self.get_user_para()
        final_ev_parameters = self.get_ev_para()
        print("提取到的最终用户偏好参数: ", final_user_parameters)
        print("提取到的最终EV参数: ", final_ev_parameters)
        return final_user_parameters, final_ev_parameters
    


    def extract_closest_data_piece(self,query):
            # 读取数据，切块
            loader = CSVLoader(file_path='/Users/feiou/llmopt/ev_agents/update_model/evbrand.csv')
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, add_start_index=True
            )
            all_splits = text_splitter.split_documents(data)

            # 构建vector数据库
            model_name = "BAAI/bge-large-zh-v1.5"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            bgeEmbeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )

            vector = Chroma.from_documents(documents=all_splits, embedding=bgeEmbeddings)

            # 提取最接近的数据piece
            retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 1})
            closest_data_piece = retriever.get_relevant_documents(query)

            return closest_data_piece[0].page_content
    
    


if __name__ == "__main__":
    
    
    input_query = "我想让我的车xiaomisu7max充电大约充8小时,我的预算不多，大概100元"
    agenti = InfoCollector(input_query)
    result = agenti.operate()