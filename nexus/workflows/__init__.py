"""Agentic workflows as LangGraph state graphs (spec 7): helpdesk, triage, pr_review, incident, change_risk.

Human gates use `langgraph.types.interrupt`; the API resumes a thread with `Command(resume=...)` after an
approver decision. State is checkpointed in SQLite (var/checkpoints.db); production uses the Postgres saver.
"""
