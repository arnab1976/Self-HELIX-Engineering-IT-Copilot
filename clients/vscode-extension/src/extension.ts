// HELIX VS Code extension: inline completion with debounce + cancellation, and an "Ask HELIX" command.
// Completion requests go to POST /v1/complete (fixed route to the T1a FIM pool, p95 <= 300 ms first token).
import * as vscode from "vscode";

let token: string | undefined;
let inflight: AbortController | undefined;
const output = vscode.window.createOutputChannel("HELIX");

function cfg<T>(key: string): T {
  return vscode.workspace.getConfiguration("helix").get<T>(key) as T;
}

async function getToken(): Promise<string> {
  if (token) return token;
  // Dev only: mint a token for a persona. Production: OIDC device-code / PKCE flow against Okta or Entra ID.
  const r = await fetch(`${cfg<string>("apiUrl")}/v1/dev/token`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ user: cfg<string>("devUser") }),
  });
  token = ((await r.json()) as { access_token: string }).access_token;
  return token;
}

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`${cfg<string>("apiUrl")}${path}`, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${await getToken()}` },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => { clearTimeout(t); reject(new Error("aborted")); });
  });

class HelixCompletionProvider implements vscode.InlineCompletionItemProvider {
  async provideInlineCompletionItems(doc: vscode.TextDocument, pos: vscode.Position, _ctx: vscode.InlineCompletionContext,
    cancel: vscode.CancellationToken): Promise<vscode.InlineCompletionItem[]> {
    // Cancel the previous request: every keystroke supersedes the last one (halves load at peak).
    inflight?.abort();
    const ctrl = new AbortController();
    inflight = ctrl;
    cancel.onCancellationRequested(() => ctrl.abort());
    try {
      await sleep(cfg<number>("debounceMs"), ctrl.signal);           // debounce
      const offset = doc.offsetAt(pos);
      const text = doc.getText();
      const res = await post<{ completion: string; latency_ms: number; model: string; trace_id: string }>("/v1/complete", {
        repo: cfg<string>("repo"), path: vscode.workspace.asRelativePath(doc.uri), language: doc.languageId,
        prefix: text.slice(Math.max(0, offset - 8000), offset), suffix: text.slice(offset, offset + 2000), max_tokens: 64,
      }, ctrl.signal);
      if (!res.completion) return [];
      output.appendLine(`completion ${res.latency_ms} ms ${res.model} ${res.trace_id}`);
      const item = new vscode.InlineCompletionItem(res.completion, new vscode.Range(pos, pos));
      item.command = { title: "accepted", command: "helix.accepted", arguments: [res.trace_id, res.completion] };
      return [item];
    } catch {
      return [];                                                       // never block typing
    }
  }
}

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, new HelixCompletionProvider()),
    vscode.commands.registerCommand("helix.accepted", async (traceId: string, completion: string) => {
      try { await post("/v1/feedback", { trace_id: traceId, signal: "accept", target: "completion", payload: { completion } }); } catch { /* best effort */ }
    }),
    vscode.commands.registerCommand("helix.signIn", async () => { token = undefined; await getToken(); vscode.window.showInformationMessage("HELIX: signed in"); }),
    vscode.commands.registerCommand("helix.ask", async () => {
      const editor = vscode.window.activeTextEditor;
      const selection = editor?.document.getText(editor.selection) ?? "";
      const q = await vscode.window.showInputBox({ prompt: "Ask HELIX about your codebase", value: selection ? `Explain: ${selection.slice(0, 200)}` : "" });
      if (!q) return;
      const res = await post<{ answer: string; citations: string[]; route: { tier: string } }>("/v1/chat", {
        messages: [{ role: "user", content: q }], repo: cfg<string>("repo"), surface: "ide",
      });
      output.clear();
      output.appendLine(res.answer);
      output.appendLine("\nCitations:\n" + res.citations.join("\n"));
      output.appendLine(`\nrouted to ${res.route.tier}`);
      output.show(true);
    }),
  );
}

export function deactivate() { inflight?.abort(); }
