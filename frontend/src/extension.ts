import * as vscode from "vscode";

type Event<T> = vscode.Event<T>;

interface RepositoryState {
  readonly HEAD:
    | { readonly name?: string; readonly commit?: string }
    | undefined;
  readonly onDidChange: Event<void>;
}

interface Repository {
  readonly rootUri: vscode.Uri;
  readonly state: RepositoryState;
  readonly onDidCommit: Event<void>;
}

type APIState = "uninitialized" | "initialized";

interface GitAPI {
  readonly state: APIState;
  readonly onDidChangeState: Event<APIState>;
  readonly repositories: Repository[];
  readonly onDidOpenRepository: Event<Repository>;
}

interface GitExtensionExports {
  getAPI(version: 1): GitAPI;
}

let panel: vscode.WebviewPanel | undefined;
let commitCount = 0;
let hookedRepos = new WeakSet<Repository>();

function ensurePanel(context: vscode.ExtensionContext) {
  if (panel) return panel;
  panel = vscode.window.createWebviewPanel(
    "commitToggleOnCommit.panel",
    "Commit Toggle Panel",
    vscode.ViewColumn.Beside,
    { enableScripts: false, retainContextWhenHidden: true }
  );
  panel.onDidDispose(() => (panel = undefined), null, context.subscriptions);
  return panel;
}

function setHtml(repoPath?: string) {
  const time = new Date().toLocaleString();
  return `<!doctype html>
<html><body style="font-family: -apple-system, Segoe UI, sans-serif; padding: 16px;">
  <h2>✅ Commit captured</h2>
  <p><b>Count</b>: ${commitCount}</p>
  <p><b>Time</b>: ${time}</p>
  <p><b>Repo</b>: <code>${repoPath ?? "(unknown)"}</code></p>
  <p style="opacity:0.7;">Note: onDidCommit typically fires for commits made via VS Code Git UI.</p>
</body></html>`;
}

function showPanel(context: vscode.ExtensionContext, repo?: Repository) {
  const p = ensurePanel(context);
  commitCount += 1;
  p.webview.html = setHtml(repo?.rootUri.fsPath);
  p.reveal(vscode.ViewColumn.Beside, true);
}

async function getGitApi(): Promise<GitAPI | undefined> {
  const ext = vscode.extensions.getExtension<GitExtensionExports>("vscode.git");
  if (!ext) return undefined;
  if (!ext.isActive) await ext.activate();
  return ext.exports?.getAPI?.(1);
}

function hookRepo(context: vscode.ExtensionContext, repo: Repository) {
  if (hookedRepos.has(repo)) return;
  hookedRepos.add(repo);

  // ✅ 핵심: repo.onDidCommit
  context.subscriptions.push(
    repo.onDidCommit(() => {
      showPanel(context, repo);
    })
  );
}

function hookAllRepos(context: vscode.ExtensionContext, git: GitAPI) {
  for (const repo of git.repositories) hookRepo(context, repo);
}

export async function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.commands.registerCommand("commitToggleOnCommit.togglePanel", () =>
      showPanel(context)
    )
  );

  const git = await getGitApi();
  if (!git) {
    vscode.window.showWarningMessage(
      "Built-in Git extension (vscode.git) not available."
    );
    return;
  }

  // ✅ Git API가 initialized 된 뒤에 repo 훅 걸기
  const init = () => {
    hookAllRepos(context, git);
    context.subscriptions.push(
      git.onDidOpenRepository((repo) => hookRepo(context, repo))
    );
  };

  if (git.state === "initialized") {
    init();
  } else {
    context.subscriptions.push(
      git.onDidChangeState((s) => {
        if (s === "initialized") init();
      })
    );
  }
}

export function deactivate() {}
