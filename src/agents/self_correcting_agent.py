"""
Self-Correcting Agent implementation using the Reflexion pattern.
This agent uses a thought-critique-refinement cycle to ensure high-quality outputs.
"""

import operator
from typing import Annotated, List, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """The state of the self-correcting agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    critique: str
    refinement_count: int
    max_refinements: int


class ReflexionAgent:
    """
    An autonomous agent that implements the Reflexion pattern for self-correction.
    It iteratively refines its output based on self-critique.
    """

    def __init__(self, model_name: str = "gpt-4-turbo-preview", max_refinements: int = 3):
        """
        Initialize the ReflexionAgent.

        Args:
            model_name: The name of the LLM to use.
            max_refinements: Maximum number of refinement cycles.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        self.max_refinements = max_refinements
        self.graph = self._build_graph()

    def _initial_generation(self, state: AgentState) -> dict:
        """
        Generates the initial response to the user's prompt.
        """
        response = self.llm.invoke(state["messages"])
        return {"messages": [response], "refinement_count": 0}

    def _critique_step(self, state: AgentState) -> dict:
        """
        Critiques the latest generated response.
        """
        last_message = state["messages"][-1].content
        critique_prompt = (
            f"Critique the following response for accuracy, completeness, and clarity. "
            f"Identify any errors or areas for improvement:\n\n{last_message}"
        )
        critique = self.llm.invoke([HumanMessage(content=critique_prompt)])
        return {"critique": critique.content}

    def _refinement_step(self, state: AgentState) -> dict:
        """
        Refines the response based on the critique.
        """
        last_message = state["messages"][-1].content
        critique = state["critique"]
        refine_prompt = (
            f"Original Response: {last_message}\n\n"
            f"Critique: {critique}\n\n"
            f"Please provide an improved version of the response incorporating the critique."
        )
        refined_response = self.llm.invoke([HumanMessage(content=refine_prompt)])
        return {
            "messages": [refined_response],
            "refinement_count": state["refinement_count"] + 1
        }

    def _should_continue(self, state: AgentState) -> str:
        """
        Determines whether to continue refinement or end the process.
        """
        if state["refinement_count"] >= state["max_refinements"]:
            return END
        
        # Logic to decide if the critique suggests no more refinement is needed
        if "no improvements needed" in state["critique"].lower() or "excellent" in state["critique"].lower():
             return END
             
        return "refine"

    def _build_graph(self) -> StateGraph:
        """
        Constructs the LangGraph for the Reflexion pattern.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("generate", self._initial_generation)
        workflow.add_node("critique", self._critique_step)
        workflow.add_node("refine", self._refinement_step)

        workflow.set_entry_point("generate")

        workflow.add_edge("generate", "critique")
        workflow.add_conditional_edges(
            "critique",
            self._should_continue,
            {
                "refine": "refine",
                END: END
            }
        )
        workflow.add_edge("refine", "critique")

        return workflow.compile()

    def run(self, prompt: str) -> str:
        """
        Executes the self-correcting workflow for a given prompt.

        Args:
            prompt: The user's input prompt.

        Returns:
            The final, refined response.
        """
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "critique": "",
            "refinement_count": 0,
            "max_refinements": self.max_refinements
        }
        
        final_state = self.graph.invoke(initial_state)
        return final_state["messages"][-1].content


if __name__ == "__main__":
    # Example usage
    agent = ReflexionAgent()
    result = agent.run("Write a brief summary of quantum computing for a CEO.")
    print(f"Final Result:\n{result}")
