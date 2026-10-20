"""HELIX Slack bot (Slack Bolt, Socket Mode): chat with HELIX and approve actions with buttons.

    pip install slack-bolt
    export SLACK_BOT_TOKEN=xoxb-... SLACK_APP_TOKEN=xapp-... HELIX_API=http://localhost:8000
    python clients/chat-bot/slack_bot.py

Slack user -> HELIX persona mapping is a dict here; production resolves the Slack user's email to the
OIDC identity and calls NEXUS with that user's token (on-behalf-of), never a shared service account.
"""
from __future__ import annotations

import os

import httpx
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

API = os.environ.get("HELIX_API", "http://localhost:8000")
USER_MAP = {}          # {"U012ABC": "sre.anita"}; default persona below
DEFAULT_PERSONA = os.environ.get("HELIX_DEFAULT_PERSONA", "u.priya")
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
_tokens: dict[str, str] = {}


def token_for(slack_user: str) -> str:
    persona = USER_MAP.get(slack_user, DEFAULT_PERSONA)
    if persona not in _tokens:
        _tokens[persona] = httpx.post(f"{API}/v1/dev/token", json={"user": persona}).json()["access_token"]
    return _tokens[persona]


def call(slack_user: str, path: str, body: dict) -> dict:
    r = httpx.post(f"{API}{path}", json=body, headers={"Authorization": f"Bearer {token_for(slack_user)}"}, timeout=60)
    return r.json() if r.status_code < 400 else {"answer": f"HELIX error {r.status_code}: {r.text[:300]}"}


@app.event("app_mention")
@app.event("message")
def on_message(event, say):
    if event.get("bot_id") or not event.get("text"):
        return
    user = event["user"]
    res = call(user, "/v1/chat", {"messages": [{"role": "user", "content": event["text"]}], "surface": "chat"})
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": res.get("answer", "")[:2900]}}]
    intr = res.get("interrupt") or {}
    if intr.get("type") == "step_up":
        blocks.append({"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "I approved the push"}, "style": "primary",
             "action_id": "step_up_ok", "value": res["thread_id"]},
            {"type": "button", "text": {"type": "plain_text", "text": "That wasn't me"}, "style": "danger",
             "action_id": "step_up_no", "value": res["thread_id"]}]})
    if res.get("citations"):
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Sources: " + ", ".join(res["citations"][:4])}]})
    say(blocks=blocks, text=res.get("answer", ""))


@app.action("step_up_ok")
@app.action("step_up_no")
def on_step_up(ack, body, say):
    ack()
    res = call(body["user"]["id"], "/v1/helpdesk/step-up",
               {"thread_id": body["actions"][0]["value"], "approved": body["actions"][0]["action_id"] == "step_up_ok"})
    say(res.get("answer", ""))


def post_approval_request(channel: str, action: dict) -> None:
    """Called by an incident notifier: interactive approval message bound to one action id."""
    app.client.chat_postMessage(channel=channel, text=f"Approval needed: {action['tool']}", blocks=[
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{action['tool']}*\n```{action['params']}```"}},
        {"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "Approve"}, "style": "primary", "action_id": "approve", "value": action["id"]},
            {"type": "button", "text": {"type": "plain_text", "text": "Reject"}, "style": "danger", "action_id": "reject", "value": action["id"]}]}])


@app.action("approve")
@app.action("reject")
def on_approval(ack, body, say):
    ack()
    aid = body["actions"][0]["value"]
    res = call(body["user"]["id"], f"/v1/actions/{aid}/approve", {"decision": body["actions"][0]["action_id"]})
    say(f"{aid}: {res.get('status', res)}")


if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
