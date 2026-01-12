import * as vscode from "vscode";
import * as cp from "child_process";

// ============================================================================
// Types
// ============================================================================

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

// Backend API Types (matching backend/domain/schemas/review.py)
type Severity = "blocker" | "high" | "medium" | "low";
type RiskLevel = "low" | "medium" | "high";
type Category =
  | "correctness"
  | "security"
  | "performance"
  | "reliability"
  | "api-compat"
  | "maintainability"
  | "style"
  | "testing"
  | "docs";

interface Location {
  file: string;
  line_start: number;
  line_end: number;
}

interface Issue {
  id: string;
  title: string;
  severity: Severity;
  category: Category;
  description: string;
  why_it_matters: string;
  suggested_fix: string;
  locations: Location[];
  confidence: number;
}

interface Summary {
  intent: string;
  overall_risk: RiskLevel;
  key_points: string[];
}

interface Meta {
  variant_id: string;
  run_id: string;
  model: string;
  generated_at: string;
}

interface PatchSuggestion {
  file: string;
  unified_diff: string;
  rationale: string;
}

interface ReviewResult {
  meta: Meta;
  summary: Summary;
  issues: Issue[];
  test_suggestions: { title: string; rationale: string }[];
  questions_to_author: { question: string; reason: string }[];
  merge_blockers: string[];
  patch_suggestions: PatchSuggestion[];
}

interface ReviewRequest {
  diff: string;
  variant_id: string;
}

// Parsed diff structure for code snippets
interface DiffHunk {
  file: string;
  oldStart: number;
  newStart: number;
  lines: { type: "context" | "add" | "remove"; content: string; lineNum: number }[];
}

// ============================================================================
// Diff Parser
// ============================================================================

function parseDiff(diffText: string): Map<string, DiffHunk[]> {
  const files = new Map<string, DiffHunk[]>();
  const lines = diffText.split("\n");

  let currentFile = "";
  let currentHunk: DiffHunk | null = null;
  let newLineNum = 0;

  for (const line of lines) {
    // Match file header: diff --git a/path b/path or +++ b/path
    const fileMatch = line.match(/^\+\+\+ b\/(.+)$/);
    if (fileMatch) {
      currentFile = fileMatch[1];
      if (!files.has(currentFile)) {
        files.set(currentFile, []);
      }
      continue;
    }

    // Match hunk header: @@ -old,count +new,count @@
    const hunkMatch = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
    if (hunkMatch && currentFile) {
      currentHunk = {
        file: currentFile,
        oldStart: parseInt(hunkMatch[1], 10),
        newStart: parseInt(hunkMatch[2], 10),
        lines: [],
      };
      newLineNum = currentHunk.newStart;
      files.get(currentFile)!.push(currentHunk);
      continue;
    }

    // Parse diff lines
    if (currentHunk && (line.startsWith("+") || line.startsWith("-") || line.startsWith(" "))) {
      const type: "add" | "remove" | "context" =
        line.startsWith("+") ? "add" :
        line.startsWith("-") ? "remove" : "context";

      currentHunk.lines.push({
        type,
        content: line.slice(1),
        lineNum: type === "remove" ? -1 : newLineNum,
      });

      if (type !== "remove") {
        newLineNum++;
      }
    }
  }

  return files;
}

function getCodeSnippetForLocation(
  parsedDiff: Map<string, DiffHunk[]>,
  location: Location
): DiffHunk | null {
  const hunks = parsedDiff.get(location.file);
  if (!hunks) return null;

  // Find the hunk that contains the specified line range
  for (const hunk of hunks) {
    const hunkStart = hunk.newStart;
    const hunkEnd = hunkStart + hunk.lines.filter(l => l.type !== "remove").length;

    if (location.line_start >= hunkStart && location.line_start <= hunkEnd) {
      return hunk;
    }
  }

  return null;
}

// ============================================================================
// State
// ============================================================================

let panel: vscode.WebviewPanel | undefined;
let hookedRepos = new WeakSet<Repository>();
let currentReviewAbortController: AbortController | undefined;

// ============================================================================
// Configuration
// ============================================================================

function getConfig() {
  const config = vscode.workspace.getConfiguration("codeReviewBot");
  return {
    apiUrl: config.get<string>("apiUrl") || "http://localhost:8000",
    variantId: config.get<string>("variantId") || "g1-mapreduce",
    autoReviewOnCommit: config.get<boolean>("autoReviewOnCommit") ?? true,
  };
}

// ============================================================================
// Git Utilities
// ============================================================================

function execGit(cwd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    cp.execFile(
      "git",
      args,
      { cwd, maxBuffer: 10 * 1024 * 1024 },
      (error, stdout, stderr) => {
        if (error) {
          reject(new Error(stderr || error.message));
        } else {
          resolve(stdout);
        }
      }
    );
  });
}

async function getLastCommitDiff(repoPath: string): Promise<string> {
  // Get diff of the last commit (HEAD~1..HEAD)
  // If there's only one commit, get the full diff of HEAD
  try {
    return await execGit(repoPath, [
      "diff",
      "HEAD~1",
      "HEAD",
      "--unified=3",
      "--no-color",
    ]);
  } catch {
    // Fallback for first commit or other issues
    return await execGit(repoPath, [
      "show",
      "HEAD",
      "--format=",
      "--unified=3",
      "--no-color",
    ]);
  }
}

async function getLastCommitInfo(
  repoPath: string
): Promise<{ hash: string; message: string; author: string }> {
  const format = await execGit(repoPath, [
    "log",
    "-1",
    "--format=%H%n%s%n%an",
  ]);
  const [hash, message, author] = format.trim().split("\n");
  return { hash: hash || "", message: message || "", author: author || "" };
}

// ============================================================================
// API Client
// ============================================================================

async function requestReview(
  diff: string,
  signal?: AbortSignal
): Promise<ReviewResult> {
  const { apiUrl, variantId } = getConfig();
  const url = `${apiUrl}/review`;

  const request: ReviewRequest = {
    diff,
    variant_id: variantId,
  };

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error (${response.status}): ${errorText}`);
  }

  return response.json() as Promise<ReviewResult>;
}

// ============================================================================
// Webview Panel
// ============================================================================

function ensurePanel(context: vscode.ExtensionContext): vscode.WebviewPanel {
  if (panel) return panel;

  panel = vscode.window.createWebviewPanel(
    "codeReviewBot.panel",
    "Code Review",
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
    }
  );

  panel.onDidDispose(() => (panel = undefined), null, context.subscriptions);
  return panel;
}

function getSeverityBadge(severity: Severity): string {
  const colors: Record<Severity, { bg: string; text: string }> = {
    blocker: { bg: "#dc2626", text: "#fff" },
    high: { bg: "#ea580c", text: "#fff" },
    medium: { bg: "#ca8a04", text: "#fff" },
    low: { bg: "#16a34a", text: "#fff" },
  };
  const c = colors[severity];
  return `<span style="background:${c.bg};color:${c.text};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;text-transform:uppercase;">${severity}</span>`;
}

function getCategoryBadge(category: Category): string {
  return `<span style="background:#374151;color:#e5e7eb;padding:2px 8px;border-radius:4px;font-size:11px;">${category}</span>`;
}

function getRiskBadge(risk: RiskLevel): string {
  const colors: Record<RiskLevel, { bg: string; text: string }> = {
    low: { bg: "#16a34a", text: "#fff" },
    medium: { bg: "#ca8a04", text: "#fff" },
    high: { bg: "#dc2626", text: "#fff" },
  };
  const c = colors[risk];
  return `<span style="background:${c.bg};color:${c.text};padding:4px 12px;border-radius:6px;font-weight:600;">${risk.toUpperCase()} RISK</span>`;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderLoadingHtml(commitInfo: {
  hash: string;
  message: string;
}): string {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      padding: 24px;
      background: #1e1e1e;
      color: #d4d4d4;
      line-height: 1.6;
    }
    .header {
      border-bottom: 1px solid #333;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    .commit-hash {
      font-family: 'SF Mono', Monaco, 'Courier New', monospace;
      color: #569cd6;
      font-size: 13px;
    }
    .commit-message {
      font-size: 18px;
      font-weight: 600;
      margin-top: 8px;
      color: #fff;
    }
    .loading {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 60px 20px;
    }
    .spinner {
      width: 48px;
      height: 48px;
      border: 4px solid #333;
      border-top-color: #569cd6;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .loading-text {
      margin-top: 20px;
      font-size: 16px;
      color: #888;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="commit-hash">${escapeHtml(commitInfo.hash.slice(0, 8))}</div>
    <div class="commit-message">${escapeHtml(commitInfo.message)}</div>
  </div>
  <div class="loading">
    <div class="spinner"></div>
    <div class="loading-text">Analyzing your code...</div>
  </div>
</body>
</html>`;
}

function renderErrorHtml(
  commitInfo: { hash: string; message: string },
  error: string
): string {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      padding: 24px;
      background: #1e1e1e;
      color: #d4d4d4;
      line-height: 1.6;
    }
    .header {
      border-bottom: 1px solid #333;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    .commit-hash {
      font-family: 'SF Mono', Monaco, 'Courier New', monospace;
      color: #569cd6;
      font-size: 13px;
    }
    .commit-message {
      font-size: 18px;
      font-weight: 600;
      margin-top: 8px;
      color: #fff;
    }
    .error-box {
      background: #3c1618;
      border: 1px solid #dc2626;
      border-radius: 8px;
      padding: 20px;
      margin: 20px 0;
    }
    .error-title {
      color: #fca5a5;
      font-weight: 600;
      margin-bottom: 8px;
    }
    .error-message {
      color: #fecaca;
      font-family: 'SF Mono', Monaco, 'Courier New', monospace;
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-all;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="commit-hash">${escapeHtml(commitInfo.hash.slice(0, 8))}</div>
    <div class="commit-message">${escapeHtml(commitInfo.message)}</div>
  </div>
  <div class="error-box">
    <div class="error-title">Review Failed</div>
    <div class="error-message">${escapeHtml(error)}</div>
  </div>
</body>
</html>`;
}

function renderCodeSnippet(hunk: DiffHunk, location: Location): string {
  // Filter lines around the issue location
  const relevantLines = hunk.lines.filter((line) => {
    if (line.lineNum === -1) return true; // Always show removed lines
    return line.lineNum >= location.line_start - 2 && line.lineNum <= location.line_end + 2;
  });

  if (relevantLines.length === 0) return "";

  const linesHtml = relevantLines
    .map((line) => {
      const lineClass =
        line.type === "add" ? "line-add" :
        line.type === "remove" ? "line-remove" : "line-context";
      const prefix = line.type === "add" ? "+" : line.type === "remove" ? "-" : " ";
      const lineNumDisplay = line.lineNum === -1 ? "  " : String(line.lineNum).padStart(3, " ");
      return `<div class="code-line ${lineClass}"><span class="line-num">${lineNumDisplay}</span><span class="line-prefix">${prefix}</span><span class="line-content">${escapeHtml(line.content)}</span></div>`;
    })
    .join("");

  return `<div class="code-snippet"><div class="code-header">${escapeHtml(hunk.file)}</div><div class="code-body">${linesHtml}</div></div>`;
}

function renderUnifiedDiff(diff: string): string {
  const lines = diff.split("\n");
  const linesHtml = lines
    .map((line) => {
      let lineClass = "line-context";
      if (line.startsWith("+") && !line.startsWith("+++")) {
        lineClass = "line-add";
      } else if (line.startsWith("-") && !line.startsWith("---")) {
        lineClass = "line-remove";
      } else if (line.startsWith("@@")) {
        lineClass = "line-hunk";
      }
      return `<div class="code-line ${lineClass}"><span class="line-content">${escapeHtml(line)}</span></div>`;
    })
    .join("");

  return `<div class="code-body">${linesHtml}</div>`;
}

function renderReviewHtml(
  commitInfo: { hash: string; message: string },
  result: ReviewResult,
  parsedDiff: Map<string, DiffHunk[]>
): string {
  const issuesHtml = result.issues
    .map((issue) => {
      // Get code snippet for the first location
      let codeSnippetHtml = "";
      if (issue.locations.length > 0) {
        const loc = issue.locations[0];
        const hunk = getCodeSnippetForLocation(parsedDiff, loc);
        if (hunk) {
          codeSnippetHtml = renderCodeSnippet(hunk, loc);
        }
      }

      return `
    <div class="issue">
      <div class="issue-header">
        ${getSeverityBadge(issue.severity)}
        ${getCategoryBadge(issue.category)}
        <span class="confidence">Confidence: ${Math.round(issue.confidence * 100)}%</span>
      </div>
      <div class="issue-title">${escapeHtml(issue.title)}</div>
      <div class="issue-description">${escapeHtml(issue.description)}</div>
      ${
        issue.locations.length > 0
          ? `<div class="issue-locations">
          ${issue.locations.map((loc) => `<code>${escapeHtml(loc.file)}:${loc.line_start}${loc.line_end !== loc.line_start ? `-${loc.line_end}` : ""}</code>`).join(" ")}
        </div>`
          : ""
      }
      ${codeSnippetHtml}
      ${
        issue.why_it_matters
          ? `<div class="issue-why">
          <strong>Why it matters:</strong> ${escapeHtml(issue.why_it_matters)}
        </div>`
          : ""
      }
      ${
        issue.suggested_fix
          ? `<div class="issue-fix">
          <strong>Suggested fix:</strong> ${escapeHtml(issue.suggested_fix)}
        </div>`
          : ""
      }
    </div>
  `;
    })
    .join("");

  const keyPointsHtml =
    result.summary.key_points.length > 0
      ? `<ul class="key-points">${result.summary.key_points.map((p) => `<li>${escapeHtml(p)}</li>`).join("")}</ul>`
      : "";

  const questionsHtml =
    result.questions_to_author.length > 0
      ? `<div class="section">
      <h3>Questions for Author</h3>
      ${result.questions_to_author.map((q) => `<div class="question"><strong>${escapeHtml(q.question)}</strong><p>${escapeHtml(q.reason)}</p></div>`).join("")}
    </div>`
      : "";

  const blockersHtml =
    result.merge_blockers.length > 0
      ? `<div class="section blockers">
      <h3>Merge Blockers</h3>
      <ul>${result.merge_blockers.map((b) => `<li>${escapeHtml(b)}</li>`).join("")}</ul>
    </div>`
      : "";

  const patchSuggestionsHtml =
    result.patch_suggestions && result.patch_suggestions.length > 0
      ? `<div class="section">
      <h3>Suggested Patches</h3>
      ${result.patch_suggestions
        .map(
          (patch) => `
        <div class="patch">
          <div class="patch-file">${escapeHtml(patch.file)}</div>
          ${patch.rationale ? `<div class="patch-rationale">${escapeHtml(patch.rationale)}</div>` : ""}
          <div class="code-snippet">
            ${renderUnifiedDiff(patch.unified_diff)}
          </div>
        </div>
      `
        )
        .join("")}
    </div>`
      : "";

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      padding: 24px;
      background: #1e1e1e;
      color: #d4d4d4;
      line-height: 1.6;
    }
    .header {
      border-bottom: 1px solid #333;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    .commit-hash {
      font-family: 'SF Mono', Monaco, 'Courier New', monospace;
      color: #569cd6;
      font-size: 13px;
    }
    .commit-message {
      font-size: 18px;
      font-weight: 600;
      margin-top: 8px;
      color: #fff;
    }
    .meta {
      font-size: 12px;
      color: #888;
      margin-top: 8px;
    }
    .summary {
      background: #252526;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
    }
    .summary-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }
    .summary-intent {
      font-size: 15px;
      color: #d4d4d4;
    }
    .key-points {
      margin: 12px 0 0 0;
      padding-left: 20px;
    }
    .key-points li {
      margin: 6px 0;
      color: #a0a0a0;
    }
    .stats {
      display: flex;
      gap: 20px;
      margin-bottom: 24px;
    }
    .stat {
      background: #252526;
      padding: 12px 20px;
      border-radius: 8px;
      text-align: center;
    }
    .stat-value {
      font-size: 24px;
      font-weight: 700;
      color: #fff;
    }
    .stat-label {
      font-size: 12px;
      color: #888;
      margin-top: 4px;
    }
    .section {
      margin-bottom: 24px;
    }
    .section h3 {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #888;
      margin-bottom: 12px;
    }
    .issue {
      background: #252526;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 12px;
      border-left: 3px solid #569cd6;
    }
    .issue-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }
    .confidence {
      margin-left: auto;
      font-size: 11px;
      color: #888;
    }
    .issue-title {
      font-size: 15px;
      font-weight: 600;
      color: #fff;
      margin-bottom: 8px;
    }
    .issue-description {
      color: #a0a0a0;
      font-size: 14px;
    }
    .issue-locations {
      margin-top: 10px;
    }
    .issue-locations code {
      background: #333;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 12px;
      margin-right: 6px;
    }
    .issue-why, .issue-fix {
      margin-top: 10px;
      font-size: 13px;
      color: #888;
    }
    .issue-why strong, .issue-fix strong {
      color: #a0a0a0;
    }
    .question {
      background: #252526;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 12px;
    }
    .question p {
      margin: 8px 0 0;
      color: #888;
    }
    .blockers {
      background: #3c1618;
      border: 1px solid #dc2626;
      border-radius: 8px;
      padding: 16px;
    }
    .blockers h3 {
      color: #fca5a5;
    }
    .blockers ul {
      margin: 0;
      padding-left: 20px;
    }
    .blockers li {
      color: #fecaca;
    }
    .no-issues {
      background: #1a3a1a;
      border: 1px solid #16a34a;
      border-radius: 8px;
      padding: 24px;
      text-align: center;
    }
    .no-issues-icon {
      font-size: 32px;
      margin-bottom: 8px;
    }
    .no-issues-text {
      color: #86efac;
      font-size: 16px;
      font-weight: 600;
    }
    /* Code snippet styles */
    .code-snippet {
      margin-top: 12px;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid #333;
    }
    .code-header {
      background: #2d2d2d;
      padding: 6px 12px;
      font-family: 'SF Mono', Monaco, 'Courier New', monospace;
      font-size: 12px;
      color: #888;
      border-bottom: 1px solid #333;
    }
    .code-body {
      background: #1a1a1a;
      font-family: 'SF Mono', Monaco, 'Courier New', monospace;
      font-size: 12px;
      line-height: 1.5;
      overflow-x: auto;
    }
    .code-line {
      display: flex;
      padding: 0 12px;
      white-space: pre;
    }
    .line-num {
      color: #555;
      min-width: 32px;
      text-align: right;
      margin-right: 8px;
      user-select: none;
    }
    .line-prefix {
      color: #888;
      min-width: 16px;
      user-select: none;
    }
    .line-content {
      flex: 1;
    }
    .line-add {
      background: rgba(46, 160, 67, 0.15);
    }
    .line-add .line-content {
      color: #7ee787;
    }
    .line-add .line-prefix {
      color: #7ee787;
    }
    .line-remove {
      background: rgba(248, 81, 73, 0.15);
    }
    .line-remove .line-content {
      color: #f85149;
    }
    .line-remove .line-prefix {
      color: #f85149;
    }
    .line-context .line-content {
      color: #8b949e;
    }
    .line-hunk {
      background: rgba(56, 139, 253, 0.1);
    }
    .line-hunk .line-content {
      color: #58a6ff;
    }
    /* Patch suggestion styles */
    .patch {
      background: #252526;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 12px;
      border-left: 3px solid #16a34a;
    }
    .patch-file {
      font-family: 'SF Mono', Monaco, 'Courier New', monospace;
      font-size: 13px;
      color: #569cd6;
      margin-bottom: 8px;
    }
    .patch-rationale {
      color: #a0a0a0;
      font-size: 13px;
      margin-bottom: 12px;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="commit-hash">${escapeHtml(commitInfo.hash.slice(0, 8))}</div>
    <div class="commit-message">${escapeHtml(commitInfo.message)}</div>
    <div class="meta">
      Reviewed with <strong>${escapeHtml(result.meta.variant_id)}</strong>
      using <strong>${escapeHtml(result.meta.model)}</strong>
      at ${escapeHtml(result.meta.generated_at)}
    </div>
  </div>

  <div class="summary">
    <div class="summary-header">
      ${getRiskBadge(result.summary.overall_risk)}
    </div>
    <div class="summary-intent">${escapeHtml(result.summary.intent)}</div>
    ${keyPointsHtml}
  </div>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">${result.issues.length}</div>
      <div class="stat-label">Issues Found</div>
    </div>
    <div class="stat">
      <div class="stat-value">${result.issues.filter((i) => i.severity === "blocker" || i.severity === "high").length}</div>
      <div class="stat-label">High Priority</div>
    </div>
    <div class="stat">
      <div class="stat-value">${result.test_suggestions.length}</div>
      <div class="stat-label">Test Suggestions</div>
    </div>
  </div>

  ${blockersHtml}

  <div class="section">
    <h3>Issues</h3>
    ${result.issues.length > 0 ? issuesHtml : '<div class="no-issues"><div class="no-issues-icon">✓</div><div class="no-issues-text">No issues found!</div></div>'}
  </div>

  ${patchSuggestionsHtml}

  ${questionsHtml}
</body>
</html>`;
}

// ============================================================================
// Main Review Flow
// ============================================================================

async function runReview(
  context: vscode.ExtensionContext,
  repo: Repository
): Promise<void> {
  const repoPath = repo.rootUri.fsPath;

  // Cancel any ongoing review
  if (currentReviewAbortController) {
    currentReviewAbortController.abort();
  }
  currentReviewAbortController = new AbortController();
  const signal = currentReviewAbortController.signal;

  try {
    // Get commit info
    const commitInfo = await getLastCommitInfo(repoPath);

    // Show loading state
    const p = ensurePanel(context);
    p.webview.html = renderLoadingHtml(commitInfo);
    p.reveal(vscode.ViewColumn.Beside, true);

    // Get diff
    const diff = await getLastCommitDiff(repoPath);

    if (!diff.trim()) {
      p.webview.html = renderErrorHtml(
        commitInfo,
        "No changes found in the last commit."
      );
      return;
    }

    // Parse diff for code snippets
    const parsedDiff = parseDiff(diff);

    // Request review
    const result = await requestReview(diff, signal);

    // Show results
    p.webview.html = renderReviewHtml(commitInfo, result, parsedDiff);
  } catch (error) {
    if ((error as Error).name === "AbortError") {
      return; // Review was cancelled
    }

    const commitInfo = await getLastCommitInfo(repoPath).catch(() => ({
      hash: "",
      message: "Unknown commit",
      author: "",
    }));

    const p = ensurePanel(context);
    p.webview.html = renderErrorHtml(
      commitInfo,
      error instanceof Error ? error.message : String(error)
    );
    p.reveal(vscode.ViewColumn.Beside, true);
  }
}

// ============================================================================
// Git Hook Setup
// ============================================================================

async function getGitApi(): Promise<GitAPI | undefined> {
  const ext =
    vscode.extensions.getExtension<GitExtensionExports>("vscode.git");
  if (!ext) return undefined;
  if (!ext.isActive) await ext.activate();
  return ext.exports?.getAPI?.(1);
}

function hookRepo(context: vscode.ExtensionContext, repo: Repository): void {
  if (hookedRepos.has(repo)) return;
  hookedRepos.add(repo);

  context.subscriptions.push(
    repo.onDidCommit(() => {
      const { autoReviewOnCommit } = getConfig();
      if (autoReviewOnCommit) {
        runReview(context, repo);
      }
    })
  );
}

function hookAllRepos(context: vscode.ExtensionContext, git: GitAPI): void {
  for (const repo of git.repositories) {
    hookRepo(context, repo);
  }
}

// ============================================================================
// Extension Lifecycle
// ============================================================================

export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "codeReviewBot.reviewLastCommit",
      async () => {
        const git = await getGitApi();
        if (!git || git.repositories.length === 0) {
          vscode.window.showWarningMessage("No Git repository found.");
          return;
        }
        await runReview(context, git.repositories[0]);
      }
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("codeReviewBot.showPanel", () => {
      const p = ensurePanel(context);
      p.reveal(vscode.ViewColumn.Beside, true);
    })
  );

  // Initialize Git hooks
  const git = await getGitApi();
  if (!git) {
    vscode.window.showWarningMessage(
      "Built-in Git extension (vscode.git) not available."
    );
    return;
  }

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

  // Show activation message
  const { apiUrl, variantId } = getConfig();
  console.log(
    `Code Review Bot activated. API: ${apiUrl}, Variant: ${variantId}`
  );
}

export function deactivate(): void {
  if (currentReviewAbortController) {
    currentReviewAbortController.abort();
  }
}
