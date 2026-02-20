"""CrewAI agent factory — creates specialized agents for each screen."""
import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool


def _get_llm(api_key: str, provider: str = "gemini") -> LLM:
    if provider == "gemini":
        os.environ["GEMINI_API_KEY"] = api_key
        return LLM(
            model="gemini-2.5-flash",
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        return LLM(model="openai/gpt-4o", api_key=api_key)


def run_agent(role: str, goal: str, backstory: str, task_description: str,
              expected_output: str, api_key: str, provider: str = "gemini",
              tools: list = None) -> str:
    """Generic agent runner — creates agent, task, crew and returns result."""
    llm = _get_llm(api_key, provider)
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools or [],
        llm=llm,
        verbose=False,
        max_iter=10,
    )
    task = Task(
        description=task_description,
        expected_output=expected_output,
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    try:
        result = crew.kickoff()
        return str(result)
    except ConnectionError as e:
        return f"⚠️ Connection error — check your API key and internet connection.\nDetails: {e}"
    except Exception as e:
        return f"⚠️ Agent error: {e}"


# ── Screen-specific agent functions ───────────────────────────────────

def parse_business_intent(data_summary: str, user_message: str,
                          api_key: str, provider: str) -> str:
    return run_agent(
        role="Business Analyst & Intent Parser",
        goal="Understand the user's business intent and define project scope from their data and requirements.",
        backstory=(
            "You are a senior business analyst. You analyze uploaded datasets and user requirements "
            "to define clear project scope, objectives, and analytics strategy. You produce structured "
            "project definitions with goals, KPIs, and recommended analytics approaches."
        ),
        task_description=(
            f"The user uploaded a dataset with this structure:\n{data_summary}\n\n"
            f"User's business requirement: {user_message}\n\n"
            f"Produce:\n"
            f"1. **Project Scope**: What this project covers\n"
            f"2. **Objectives**: 3-5 clear business objectives\n"
            f"3. **Key Questions**: What analytics questions to answer\n"
            f"4. **Recommended Approach**: Descriptive/Diagnostic/Predictive/Prescriptive\n"
            f"5. **KPIs**: Key metrics to track\n"
            f"6. **Data Assessment**: Brief assessment of data suitability"
        ),
        expected_output="A structured project scope document with objectives, questions, approach, and KPIs.",
        api_key=api_key, provider=provider,
    )


def run_diagnostic_analysis(data_summary: str, question: str,
                            api_key: str, provider: str) -> str:
    return run_agent(
        role="Diagnostic Data Analyst",
        goal="Identify root causes and correlations in the data to explain WHY something happened.",
        backstory=(
            "You are an expert diagnostic analyst. Given data summaries and a question about why "
            "something occurred, you identify correlations, root causes, contributing factors, "
            "and provide evidence-based explanations."
        ),
        task_description=(
            f"Data summary:\n{data_summary}\n\n"
            f"User's diagnostic question: {question}\n\n"
            f"Analyze the data and provide:\n"
            f"1. **Root Cause Analysis**: What caused the observed pattern\n"
            f"2. **Contributing Factors**: Other variables that played a role\n"
            f"3. **Correlations Found**: Key relationships in the data\n"
            f"4. **Evidence**: Specific data points supporting your analysis\n"
            f"5. **Recommendations**: What to investigate further"
        ),
        expected_output="A diagnostic report with root causes, correlations, and evidence.",
        api_key=api_key, provider=provider,
    )


def generate_ba_document(doc_type: str, project_scope: str, data_summary: str,
                         api_key: str, provider: str) -> str:
    templates = {
        "BRD": (
            "Generate a Business Requirements Document (BRD) including:\n"
            "1. Executive Summary\n2. Business Objectives\n3. Stakeholders\n"
            "4. Business Requirements (functional & non-functional)\n"
            "5. Success Criteria\n6. Assumptions & Constraints\n7. Risks"
        ),
        "FRD": (
            "Generate a Functional Requirements Document (FRD) including:\n"
            "1. System Overview\n2. Functional Requirements (numbered)\n"
            "3. Data Requirements\n4. Interface Requirements\n"
            "5. Processing Logic\n6. Reporting Requirements\n7. Acceptance Criteria"
        ),
        "SRS": (
            "Generate a Software Requirements Specification (SRS) including:\n"
            "1. Introduction & Purpose\n2. System Description\n"
            "3. Functional Requirements\n4. Non-Functional Requirements\n"
            "5. Data Model\n6. System Architecture\n7. Technical Constraints"
        ),
    }

    return run_agent(
        role=f"Senior Business Analyst — {doc_type} Writer",
        goal=f"Generate a professional {doc_type} document based on project scope and data.",
        backstory=f"You write enterprise-grade {doc_type} documents following industry standards.",
        task_description=(
            f"Project Scope:\n{project_scope}\n\n"
            f"Data Summary:\n{data_summary}\n\n"
            f"{templates.get(doc_type, templates['BRD'])}"
        ),
        expected_output=f"A complete, professional {doc_type} document in markdown format.",
        api_key=api_key, provider=provider,
    )


def generate_effort_estimate(project_scope: str, api_key: str, provider: str) -> str:
    return run_agent(
        role="Project Estimation Expert",
        goal="Provide AI-powered effort estimation for the analytics project.",
        backstory="You estimate project effort using industry benchmarks and historical patterns.",
        task_description=(
            f"Project Scope:\n{project_scope}\n\n"
            f"Provide:\n"
            f"1. **Phase Breakdown**: List each phase with estimated hours\n"
            f"2. **Team Composition**: Roles needed\n"
            f"3. **Timeline**: Gantt-chart-style breakdown (Phase | Start Week | End Week | Hours)\n"
            f"4. **Total Effort**: Sum of hours\n"
            f"5. **Risk Buffer**: Additional time for risks\n"
            f"Format the timeline as a markdown table."
        ),
        expected_output="Effort estimation with phase breakdown, timeline table, and team composition.",
        api_key=api_key, provider=provider,
    )


def get_model_recommendation(data_summary: str, objective: str,
                             api_key: str, provider: str) -> str:
    return run_agent(
        role="ML/AI Model Selection Expert",
        goal="Recommend the best ML/DL model for the user's objective based on their data.",
        backstory="You are a machine learning expert who recommends optimal models based on data characteristics.",
        task_description=(
            f"Data summary:\n{data_summary}\n\n"
            f"Objective: {objective}\n\n"
            f"Provide:\n"
            f"1. **Recommended Model**: Name and type (ML or DL)\n"
            f"2. **Why This Model**: Justification based on data\n"
            f"3. **Alternative Models**: 2-3 alternatives\n"
            f"4. **Feature Engineering**: Suggested transformations\n"
            f"5. **Expected Performance**: Rough accuracy range\n"
            f"6. **Hyperparameters**: Key ones to tune"
        ),
        expected_output="Model recommendation with justification, alternatives, and tuning tips.",
        api_key=api_key, provider=provider,
    )


def run_prescriptive_analysis(data_summary: str, goal_description: str,
                              api_key: str, provider: str) -> str:
    return run_agent(
        role="Prescriptive Analytics & Optimization Expert",
        goal="Provide actionable recommendations with confidence scores to achieve the user's business goal.",
        backstory=(
            "You are a prescriptive analytics expert. You simulate scenarios, "
            "run goal-seek analyses, and provide ranked recommendations with confidence levels."
        ),
        task_description=(
            f"Data summary:\n{data_summary}\n\n"
            f"Business goal: {goal_description}\n\n"
            f"Provide:\n"
            f"1. **Goal Analysis**: What needs to change to achieve the goal\n"
            f"2. **Scenario Simulations**: 3-4 scenarios with expected outcomes\n"
            f"3. **Recommendations**: Ranked actions with confidence scores (0-100%)\n"
            f"4. **Risk Assessment**: Potential downsides of each recommendation\n"
            f"5. **Implementation Steps**: Concrete operational steps\n"
            f"Format recommendations as a table: Action | Impact | Confidence | Risk"
        ),
        expected_output="Prescriptive recommendations with scenarios, confidence scores, and implementation steps.",
        api_key=api_key, provider=provider,
    )


def generate_executive_summary(context: str, audience: str,
                               api_key: str, provider: str) -> str:
    return run_agent(
        role="Executive Report Compiler",
        goal="Compile all analytics findings into a concise executive summary.",
        backstory="You write C-suite-ready executive summaries that are clear, concise, and actionable.",
        task_description=(
            f"All analytics context:\n{context}\n\n"
            f"Target audience: {audience}\n\n"
            f"Generate an executive summary tailored for **{audience}** with:\n"
            f"1. **Executive Overview** (2-3 sentences)\n"
            f"2. **Key Findings** (bullet points)\n"
            f"3. **Critical Metrics** (table format)\n"
            f"4. **Recommendations** (prioritized)\n"
            f"5. **Next Steps**"
        ),
        expected_output="A concise, executive-ready summary in markdown.",
        api_key=api_key, provider=provider,
    )
