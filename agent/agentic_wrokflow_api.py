from tools.currency_conv_tool import CurrencyConverterTool
from tools.expense_calc_tool import CalculatorTool
from utils.model_loader import ModelLoader
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from tools.weather_info_tool import WeatherInfoTool
from tools.place_search_tool import PlaceSearchTool
from prompt_library.system_prompt import SYSTEM_PROMPT

class GraphBuilder():
    def __init__(self, model_provider:str="groq"):
        print("My name is Kushagra")
        self.model_loader = ModelLoader(model_provider=model_provider)
        self.llm = self.model_loader.load_llm()
        self.tools = []
        self.weather_tools = WeatherInfoTool()
        self.currency_conv_tools=CurrencyConverterTool()
        self.expense_calculator_tools=CalculatorTool()
        self.place_search_tools=PlaceSearchTool()
        self.tools.extend([* self.weather_tools.weather_tool_list,
                            *  self.place_search_tools.place_search_tool_list,
                            * self.currency_conv_tools.currency_converter_tool_list,
                            * self.expense_calculator_tools.calculator_tool_list
                           ])
        
        self.llm_with_tools=self.llm.bind_tools(tools=self.tools)

        self.system_prompt=SYSTEM_PROMPT
        self.graph = None
        
    def agent_function(self, state:MessagesState):
        user_question=state["messages"]
        input_question=[self.system_prompt]+ user_question
        response= self.llm_with_tools.invoke(input_question)
        return {"messages": [response]}

    def build_graph(self):
        graph_builder=StateGraph(MessagesState)
        graph_builder.add_node("agent", self.agent_function)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_edge(START,"agent")
        graph_builder.add_conditional_edges("agent",tools_condition)
        graph_builder.add_edge("tools","agent")
        graph_builder.add_edge("agent",END)
        self.graph = graph_builder.compile()
        return self.graph

    def __call__(self):
        return self.build_graph()

