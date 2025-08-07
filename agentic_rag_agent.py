   # Do thời gian có hạn , nên em sẽ triển khai trước với 1 pain point cụ thể mà sẽ chưa có tính năng quy mô doanh nghiệp
import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import StateGraph, END

os.environ["GOOGLE_API_KEY"] = ""

class AgentState(TypedDict):
    pain_point: str
    deconstructed: Optional[Dict[str, List[str]]]
    best_feature: Optional[Dict[str, str]]  
    selected_features: List[Dict[str, Any]]  
    evaluation_results: Optional[Dict[str, Any]]
    resolution_tracking: Optional[Dict[str, Any]]
    alternative_suggestions: Optional[List[Dict[str, Any]]]
    summary: Optional[Dict[str, Any]]
    current_step: str
    error: Optional[str]
    thoughts: List[str]
    actions: List[Dict[str, Any]]
    observations: List[str]
    should_continue_search: bool
    search_iteration: int
    max_iterations: int
    unresolved_problems: List[str]
    final_output: Optional[Dict[str, Any]]
    current_search_term: Optional[str]
    candidate_features: List[Dict[str, Any]]
    is_valid_question: Optional[bool]
    validation_message: Optional[str]

class EnhancedFeatureResolutionAgent:
    def __init__(self, max_search_iterations: int = 3):
        self.setup_vector_store()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0
        )
        self.max_iterations = max_search_iterations
        self.workflow = self.create_workflow()
    
    def setup_vector_store(self):
        print("Setting up vector store...")
        """Initialize the vector store with feature data"""
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
        #Tạo document objects from dataset
        documents = []
        for item in dataset:
            page_content = (
                f"Feature Name: {item['feature_name']}\n"
                f"Description: {item['description']}\n"
                f"Keywords: {', '.join(item['keywords'])}\n"
                f"Pain Points Solved: {', '.join(item['pain_points_solved'])}\n"
                f"Link: {item['link']}"
            )
            
            metadata = {
                "feature_id": item["feature_id"],
                "feature_name": item["feature_name"],
                "link": item["link"]
            }
            
            documents.append(Document(
                page_content=page_content,
                metadata=metadata
            ))
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory="./chroma_db"
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}  
        )
    
    def validate_question_node(self, state: AgentState) -> AgentState:
        print("Validating question...")
        """Node to validate if the user's question is business-related"""
        try:
            pain_point = state["pain_point"]
            thought = f"Validating if question is business-related: {pain_point}"
            
            parser = JsonOutputParser(pydantic_object=None)
            
            prompt_template = ChatPromptTemplate.from_template(
                "You are a business solution recommendation agent. Your purpose is to help businesses find software features and solutions for their operational challenges.\n\n"
                "EVALUATE if the user's question/request is business-related and appropriate for this agent.\n\n"
                "**VALID questions include**:\n"
                "- Business operational problems (e.g., 'We struggle with customer feedback collection')\n"
                "- Workflow inefficiencies (e.g., 'Our team has trouble tracking project progress')\n"
                "- Business process challenges (e.g., 'We need better inventory management')\n"
                "- Team collaboration issues (e.g., 'Communication between departments is poor')\n"
                "- Customer service problems (e.g., 'We can't respond to customer inquiries quickly')\n"
                "- Data management issues (e.g., 'We lose track of important customer data')\n"
                "- Business growth challenges (e.g., 'We need to scale our operations')\n\n"
                "**INVALID questions include**:\n"
                "- Personal requests (e.g., 'I want to eat candy', 'What's the weather like?')\n"
                "- Non-business topics (e.g., 'How to cook pasta', 'Tell me a joke')\n"
                "- Academic questions (e.g., 'What is photosynthesis?')\n"
                "- Entertainment requests (e.g., 'Recommend a movie')\n"
                "- Personal health/medical questions\n"
                "- General knowledge questions unrelated to business\n\n"
                "**User's Question**: {pain_point}\n\n"
                "**Instructions**:\n"
                "1. Determine if this is a business-related question\n"
                "2. Check if it's asking for business solutions, operational improvements, or process optimization\n"
                "3. Output JSON with exactly these keys:\n"
                "   - 'is_valid': true/false\n"
                "   - 'reason': brief explanation of why valid/invalid\n"
                "   - 'business_relevance_score': 0-1 (1 = highly business relevant)\n\n"
                "Output ONLY valid JSON."
            )
            
            chain = prompt_template | self.llm | parser
            result = chain.invoke({"pain_point": pain_point})
            
            is_valid = result.get('is_valid', False)
            reason = result.get('reason', 'No reason provided')
            relevance_score = result.get('business_relevance_score', 0)
            
            observation = f"Question validation: {'Valid' if is_valid else 'Invalid'} - {reason}"
            print(f"Question validation result: {result}")
            
            if not is_valid:
                validation_message = (
                    f"Tôi không thể giúp bạn với câu hỏi này. "
                    f"Đây là một agent chuyên về gợi ý tính năng và giải pháp doanh nghiệp. "
                    f"Vui lòng hỏi về các vấn đề liên quan đến hoạt động kinh doanh, "
                    f"quy trình làm việc, hoặc thách thức trong doanh nghiệp của bạn.\n\n"
                    f"Lý do: {reason}"
                )
            else:
                validation_message = None
            
            return {
                **state,
                "is_valid_question": is_valid,
                "validation_message": validation_message,
                "current_step": "validated",
                "thoughts": state.get("thoughts", []) + [thought],
                "observations": state.get("observations", []) + [observation]
            }
        except Exception as e:
            return {
                **state,
                "error": f"Question validation failed: {str(e)}",
                "current_step": "error"
            }
    
    def deconstruct_pain_point_node(self, state: AgentState) -> AgentState:
        print("Deconstructing pain point...")
        """Node to deconstruct pain point into problems and outcomes"""
        try:
            thought = f"Analyzing pain point: {state['pain_point']}"
            
            parser = JsonOutputParser(pydantic_object=None)
            
            prompt_template = ChatPromptTemplate.from_template(
                "Analyze the following customer pain point description and extract:\n"
                "1. CURRENT problems the user is experiencing (list of strings)\n"
                "2. DESIRED outcomes they want to achieve (list of strings)\n\n"
                "Format output as JSON with exactly these keys: 'current_problems', 'desired_outcomes'\n\n"
                "Pain point description: {pain_point}\n\n"
                "Guidelines:\n"
                "- Return empty lists if no relevant items found\n"
                "- Never add explanations or additional text\n"
                "- Keep items concise (3-5 words each)\n"
                "- Extract verbatim phrases when possible\n"
            )
            
            chain = prompt_template | self.llm | parser
            result = chain.invoke({"pain_point": state["pain_point"]})
            
            observation = f"Deconstructed into {len(result.get('current_problems', []))} problems and {len(result.get('desired_outcomes', []))} outcomes"
            print("Deconstruction result:", result)
            
            return {
                **state,
                "deconstructed": result,
                "current_step": "deconstructed",
                "thoughts": state.get("thoughts", []) + [thought],
                "observations": state.get("observations", []) + [observation]
            }
        except Exception as e:
            return {
                **state,
                "error": f"Deconstruction failed: {str(e)}",
                "current_step": "error"
            }
    
    def match_features_node(self, state: AgentState) -> AgentState:
        print("Matching features...")
        """Node to match features to a single problem/outcome"""
        try:
            deconstructed = state["deconstructed"]
            if state.get("search_iteration", 0) > 0 and state.get("unresolved_problems"):
                search_term = state["unresolved_problems"][0]
                thought = f"Searching for features to solve: {search_term} (iteration {state.get('search_iteration', 1)})"
            else:
                all_terms = deconstructed["current_problems"] + deconstructed["desired_outcomes"]
                if all_terms:
                    search_term = all_terms[0]
                    thought = f"Searching for features to solve: {search_term}"
                else:
                    return {
                        **state,
                        "error": "No problems or outcomes found to search for",
                        "current_step": "error"
                    }
            try:
                docs = self.retriever.invoke(search_term)
                candidate_features = []
                for doc in docs:
                    candidate_features.append({
                        "feature_id": doc.metadata["feature_id"],
                        "feature_name": doc.metadata["feature_name"],
                        "context": doc.page_content,
                        "link": doc.metadata["link"],
                        "source": "langchain_enhanced"
                    })
                
            except Exception as e:
                candidate_features = []
            
            observation = f"Found {len(candidate_features)} candidate features for: {search_term}"
            print(f"Found {len(candidate_features)} candidate features for:")
            return {
                **state,
                "current_search_term": search_term,
                "candidate_features": candidate_features,
                "current_step": "matched_features",
                "thoughts": state.get("thoughts", []) + [thought],
                "observations": state.get("observations", []) + [observation]
            }
        except Exception as e:
            return {
                **state,
                "error": f"Feature matching failed: {str(e)}",
                "current_step": "error"
            }
    
    def select_best_feature_node(self, state: AgentState) -> AgentState:
        print("Selecting best feature...")
        """Node to select the best feature for the current search term"""
        try:
            candidate_features = state.get("candidate_features", [])
            search_term = state["current_search_term"]
            deconstructed = state["deconstructed"]
            selected_features = state.get("selected_features", [])
            
            thought = f"Selecting best feature to solve: {search_term}"
            
            if not candidate_features:
                print("no candidate features found in select_best_feature_node")
                return {
                    **state,
                    "best_feature": None,
                    "current_step": "no_feature_selected",
                    "thoughts": state.get("thoughts", []) + [thought],
                    "observations": state.get("observations", []) + ["No suitable features found"]
                }

            candidates_str = "\n\n".join([
                f"Feature ID: {c['feature_id']}\n"
                f"Name: {c['feature_name']}\n"
                f"Link: {c['link']}\n"
                "Details:\n{c['context']}"
                for c in candidate_features
            ])
            
            prompt = ChatPromptTemplate.from_template(
                "Select the single best feature that addresses the specific problem/outcome.\n\n"
                "**Problem/Outcome to Solve**:\n{search_term}\n\n"
                "**Customer Pain Point**:\n{pain_point}\n\n"
                "**Already Selected Features**:\n{selected_features_str}\n\n"
                "**Candidate Features**:\n{candidates}\n\n"
                "**Instructions**:\n"
                "1. Choose ONE feature that best solves this specific problem/outcome\n"
                "2. Consider relevance and effectiveness\n"
                "3. Avoid selecting features already chosen\n"
                "4. Output JSON with: 'feature_id', 'feature_name', 'link', 'reason', and 'confidence_score'\n"
                "5. 'reason' should be a concise 1-2 sentence explanation\n"
                "6. 'confidence_score' should be 0-1 (1 = perfect match)\n"
                "7. If no good match exists, set feature_id to null\n\n"
                "Output ONLY valid JSON."
            )

            selected_features_str = "\n".join([
                f"- {f.get('feature_name', 'Unknown')}: {f.get('reason', 'No reason provided')}"
                for f in selected_features
            ]) if selected_features else "None"
            
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({
                "search_term": search_term,
                "pain_point": state["pain_point"],
                "selected_features_str": selected_features_str,
                "candidates": candidates_str
            })
            
            observation = f"Selected feature: {result.get('feature_name', 'None')} with confidence {result.get('confidence_score', 0)}"
            print("best feature in select_best_feature_node is", result)
            
            # Add the selected feature to the list if it's valid
            if result.get("feature_id"):
                selected_features.append(result)
            
            return {
                **state,
                "best_feature": result if result.get("feature_id") else None,
                "selected_features": selected_features,
                "current_step": "feature_selected",
                "thoughts": state.get("thoughts", []) + [thought],
                "observations": state.get("observations", []) + [observation]
            }
        except Exception as e:
            return {
                **state,
                "error": f"Feature selection failed: {str(e)}",
                "current_step": "error"
            }
    
    def evaluate_outcomes_node(self, state: AgentState) -> AgentState:
        print("Evaluating outcomes...")
        """Node to evaluate how well the feature addresses the user's actual problems and outcomes"""
        try:
            best_feature = state["best_feature"]
            deconstructed = state["deconstructed"]
            search_term = state["current_search_term"]
            
            thought = "Evaluating how well the selected feature addresses the user's needs"
            
            if best_feature is None:
                return {
                    **state,
                    "evaluation_results": {
                        "overall_status": "no_feature_recommended",
                        "problem_resolution": [{"problem": p, "status": "not_resolved"} for p in deconstructed["current_problems"]],
                        "outcome_achievement": [{"outcome": o, "status": "not_achieved"} for o in deconstructed["desired_outcomes"]]
                    },
                    "current_step": "evaluated",
                    "thoughts": state.get("thoughts", []) + [thought],
                    "observations": state.get("observations", []) + ["No feature to evaluate"]
                }
            feature_id = best_feature["feature_id"]
            try:
                docs = self.retriever.invoke(f"Feature ID: {feature_id}")
                feature_context = docs[0].page_content if docs else "No context available"
            except Exception:
                feature_context = "No context available"
            
            prompt_template = ChatPromptTemplate.from_template(
                "EVALUATE how well the recommended feature addresses the user's ACTUAL problems and desired outcomes.\n\n"
                "**Recommended Feature**\n"
                "ID: {feature_id}\n"
                "Name: {feature_name}\n"
                "Link: {link}\n"
                "Details:\n{feature_context}\n\n"
                "**User's Current Problems**\n{problems}\n\n"
                "**User's Desired Outcomes**\n{outcomes}\n\n"
                "**Evaluation Tasks**\n"
                "1. For EACH current problem:\n"
                "   - Determine if feature Does or DOES NOT resolve it\n"
                "   - Provide 1-sentence technical justification\n"
                "   - Rate resolution effectiveness (0-1)\n"
                "2. For EACH desired outcome:\n"
                "   - Determine if feature DOES or DOES NOT achieve it\n"
                "   - Provide 1-sentence technical justification\n"
                "   - Rate achievement effectiveness (0-1)\n"
                "3. Give overall resolution status\n\n"
                "**Output Format**\n"
                "{{"
                "\"overall_status\": \"resolved\"  | \"partially_resolved\" | \"not_resolved\","
                "\"problem_resolution\": ["
                "   {{\"problem\": \"text\", \"status\": \"resolved\"  | \"not_resolved\", \"reason\": \"explanation\", \"effectiveness\": 0.8}}"
                "],"
                "\"outcome_achievement\": ["
                "   {{\"outcome\": \"text\", \"status\": \"achieved\"  | \"not_achieved\", \"reason\": \"explanation\", \"effectiveness\": 0.8}}"
                "]}}"
            )
            
            chain = prompt_template | self.llm | JsonOutputParser()
            result = chain.invoke({
                "feature_id": feature_id,
                "feature_name": best_feature["feature_name"],
                "link": best_feature.get("link", ""),
                "feature_context": feature_context,
                "problems": "\n- ".join(deconstructed["current_problems"]),
                "outcomes": "\n- ".join(deconstructed["desired_outcomes"])
            })
            
            observation = f"Evaluation complete: {result.get('overall_status', 'unknown')} status"
            
            return {
                **state,
                "evaluation_results": result,
                "current_step": "evaluated",
                "thoughts": state.get("thoughts", []) + [thought],
                "observations": state.get("observations", []) + [observation]
            }
        except Exception as e:
            return {
                **state,
                "error": f"Evaluation failed: {str(e)}",
                "current_step": "error"
            }
    
    def decide_continue_search_node(self, state: AgentState) -> AgentState:
        """ReAct node: Decide whether to continue searching for more features"""
        try:
            evaluation = state["evaluation_results"]
            current_iteration = state.get("search_iteration", 0)
            deconstructed = state["deconstructed"]
            selected_features = state.get("selected_features", [])
            
            thought = f"Evaluating whether to continue search (iteration {current_iteration + 1}/{self.max_iterations})"
            
            # Check if we have enough good features
            good_features_count = len([f for f in selected_features if f.get("confidence_score", 0) >= 0.8])
            has_good_features = good_features_count > 0
            unresolved_count = 0
            low_effectiveness_count = 0
            unresolved_problems = []
            resolved_count = 0
            
            for problem_item in evaluation.get("problem_resolution", []):
                if problem_item.get("status") == "resolved":
                    resolved_count += 1
                else:
                    unresolved_count += 1
                    unresolved_problems.append(problem_item["problem"])
                if problem_item.get("effectiveness", 1) < 0.7:
                    low_effectiveness_count += 1
            
            for outcome_item in evaluation.get("outcome_achievement", []):
                if outcome_item.get("status") == "achieved":
                    resolved_count += 1
                else:
                    unresolved_count += 1
                if outcome_item.get("effectiveness", 1) < 0.7:
                    low_effectiveness_count += 1

            total_problems = len(evaluation.get("problem_resolution", [])) + len(evaluation.get("outcome_achievement", []))
            most_problems_resolved = resolved_count > 0 and (resolved_count / total_problems) >= 0.7  

            should_continue = (
                current_iteration < self.max_iterations and
                (unresolved_count > 0 or low_effectiveness_count > 0) and
                evaluation.get("overall_status") != "resolved" and
                not most_problems_resolved 
            )
            
            observation = f"Decision: {'Continue search' if should_continue else 'Stop search'} - Has good features: {has_good_features}, Most problems resolved: {most_problems_resolved}, {resolved_count}/{total_problems} resolved, {unresolved_count} unresolved"
            print("Decision made in decide_continue_search_node:", should_continue)
            return {
                **state,
                "should_continue_search": should_continue,
                "search_iteration": current_iteration + 1,
                "unresolved_problems": unresolved_problems if should_continue else [],
                "current_step": "search_decision_made",
                "thoughts": state.get("thoughts", []) + [thought],
                "observations": state.get("observations", []) + [observation]
            }
        except Exception as e:
            return {
                **state,
                "error": f"Search decision failed: {str(e)}",
                "current_step": "error"
            }
    
    def format_output_node(self, state: AgentState) -> AgentState:
        print("Formatting output...")
        """Node to format the final output as JSON with suggested solutions"""
        try:
            selected_features = state.get("selected_features", [])
            evaluation = state["evaluation_results"]
            deconstructed = state["deconstructed"]
            
            thought = "Formatting final output as structured JSON"
            
            if not selected_features:
                suggested_solutions = []
            else:
                suggested_solutions = []
                
                for feature in selected_features:
                    feature_name = feature.get("feature_name", "Unknown Feature")
                    feature_id = feature.get("feature_id", "")
                    link = feature.get("link", "")
                    prompt_template = ChatPromptTemplate.from_template(
                        "Based on the feature details and the user's pain point, explain how this feature helps solve their problem.\n\n"
                        "**User's Pain Point**:\n{pain_point}\n\n"
                        "**Feature Name**: {feature_name}\n"
                        "**Feature Details**:\n{feature_context}\n\n"
                        "Write a clear, concise explanation (2-3 sentences) of how this feature helps the user. "
                        "Focus on the immediate benefits and relief it provides."
                    )

                    try:
                        docs = self.retriever.invoke(f"Feature ID: {feature_id}")
                        feature_context = docs[0].page_content if docs else "No context available"
                    except Exception:
                        feature_context = "No context available"
                    
                    chain = prompt_template | self.llm
                    how_it_helps_result = chain.invoke({
                        "pain_point": state["pain_point"],
                        "feature_name": feature_name,
                        "feature_context": feature_context
                    })
                    
                    how_it_helps = how_it_helps_result.content.strip()
                    
                    # Extract category from feature_name (part before the dash)
                    category = feature_name.split(" - ")[0] if " - " in feature_name else "Other"
                    
                    suggested_solutions.append({
                        "feature_name": feature_name,
                        "how_it_helps": how_it_helps,
                        "relevance_score": round(feature.get("confidence_score", 0), 2),
                        "link_to_info": link,
                        "category": category
                    })
                suggested_solutions.sort(key=lambda x: x["relevance_score"], reverse=True)
        
            final_output = {
                "suggested_solutions": suggested_solutions,
                "analysis_summary": {
                    "total_problems": len(deconstructed.get("current_problems", [])),
                    "total_outcomes": len(deconstructed.get("desired_outcomes", [])),
                    "resolved_count": len([p for p in evaluation.get("problem_resolution", []) if p.get("status") == "resolved"]) + \
                                    len([o for o in evaluation.get("outcome_achievement", []) if o.get("status") == "achieved"]),
                    "search_iterations": state.get("search_iteration", 0),
                    "overall_status": evaluation.get("overall_status", "unknown"),
                    "total_features_selected": len(selected_features)
                }
            }
            
            observation = f"Formatted output with {len(suggested_solutions)} suggested solution(s)"
            
            return {
                **state,
                "final_output": final_output
            }
        except Exception as e:
            return {
                **state,
                "error": f"Output formatting failed: {str(e)}",
                "current_step": "error"
            }
    
    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with ReAct pattern"""
        workflow = StateGraph(AgentState)

        workflow.add_node("validate_question", self.validate_question_node)
        workflow.add_node("deconstruct", self.deconstruct_pain_point_node)
        workflow.add_node("match_features", self.match_features_node)
        workflow.add_node("select_feature", self.select_best_feature_node)
        workflow.add_node("evaluate", self.evaluate_outcomes_node)
        workflow.add_node("decide_continue", self.decide_continue_search_node)
        workflow.add_node("format_output", self.format_output_node)
        

        workflow.set_entry_point("validate_question")

        def is_valid_question(state):
            return state.get("is_valid_question", False)
        
        workflow.add_conditional_edges(
            "validate_question",
            is_valid_question,
            {
                True: "deconstruct",
                False: END
            }
        )
        
        workflow.add_edge("deconstruct", "match_features")
        workflow.add_edge("match_features", "select_feature")
        workflow.add_edge("select_feature", "evaluate")
        workflow.add_edge("evaluate", "decide_continue")
        
        # Conditional edge based on search decision
        def should_continue_search(state):
            return state.get("should_continue_search", False)
        
        workflow.add_conditional_edges(
            "decide_continue",
            should_continue_search,
            {
                True: "match_features",  
                False: "format_output"  
            }
        )
        
        workflow.add_edge("format_output", END)
        
        return workflow.compile()
    
    def analyze_pain_point(self, pain_point: str) -> Dict[str, Any]:
        """Main method to analyze a pain point with ReAct pattern"""
        print("Analyzing pain point...")
        initial_state = {
            "pain_point": pain_point,
            "deconstructed": None,
            "best_feature": None,
            "selected_features": [],
            "evaluation_results": None,
            "resolution_tracking": None,
            "alternative_suggestions": [],
            "summary": None,
            "current_step": "started",
            "error": None,
            "thoughts": [],
            "actions": [],
            "observations": [],
            "should_continue_search": False,
            "search_iteration": 0,
            "max_iterations": self.max_iterations,
            "unresolved_problems": [],
            "final_output": None,
            "current_search_term": None,
            "candidate_features": [],
            "is_valid_question": None,
            "validation_message": None
        }
        
        result = self.workflow.invoke(initial_state)
        return result


if __name__ == "__main__":
    agent = EnhancedFeatureResolutionAgent(max_search_iterations=1)



    # Do thời gian có hạn , nên em sẽ triển khai trước với 1 pain point cụ thể mà sẽ chưa có tính năng quy mô doanh nghiệp
    pain_point = "We struggle to collect customer feedback after purchases."




    result = agent.analyze_pain_point(pain_point)
    
    if result.get("validation_message"):
        print("VALIDATION MESSAGE:")
        print(result["validation_message"])
    else:
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(result["final_output"], f, ensure_ascii=False, indent=2)

