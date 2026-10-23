import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { expect, test } from "vitest";

import { createTestLogger } from "../../test-utils/test-logger.js";
import { AgentManager } from "./agent-manager.js";
import { ensureAgentHistoryLoaded, ensureAgentLoaded } from "./agent-loading.js";
import { startAgentRun } from "./agent-prompt.js";
import { AgentStorage } from "./agent-storage.js";
import type {
  AgentClient,
  AgentLaunchContext,
  AgentPersistenceHandle,
  AgentResumeSessionOptions,
  AgentSession,
  AgentSessionConfig,
} from "./agent-sdk-types.js";
import { createTestAgentClients } from "../test-utils/fake-agent-client.js";

test("loads archived records for history and active records with the interactive default", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "agent-loading-purpose-"));
  const logger = createTestLogger();
  const storage = new AgentStorage(path.join(root, "agents"), logger);
  const baseClient = createTestAgentClients().codex;
  if (!baseClient) {
    throw new Error("expected Codex test client");
  }

  const resumeOptions: Array<AgentResumeSessionOptions | undefined> = [];
  const client: AgentClient = {
    provider: baseClient.provider,
    capabilities: baseClient.capabilities,
    createSession: async (
      config: AgentSessionConfig,
      launchContext?: AgentLaunchContext,
    ): Promise<AgentSession> => await baseClient.createSession(config, launchContext),
    resumeSession: async (
      handle: AgentPersistenceHandle,
      overrides?: Partial<AgentSessionConfig>,
      launchContext?: AgentLaunchContext,
      options?: AgentResumeSessionOptions,
    ): Promise<AgentSession> => {
      resumeOptions.push(options);
      return await baseClient.resumeSession(handle, overrides, launchContext);
    },
    fetchCatalog: async (options) => await baseClient.fetchCatalog(options),
    isAvailable: async () => await baseClient.isAvailable(),
  };
  const manager = new AgentManager({
    clients: { codex: client },
    registry: storage,
    logger,
  });

  const archivedId = "00000000-0000-4000-8000-000000000301";
  const activeId = "00000000-0000-4000-8000-000000000302";

  try {
    const archived = await manager.createAgent({ provider: "codex", cwd: root }, archivedId, {
      workspaceId: "workspace-archived",
    });
    await manager.archiveAgent(archived.id);

    const active = await manager.createAgent({ provider: "codex", cwd: root }, activeId, {
      workspaceId: "workspace-active",
    });
    await manager.closeAgent(active.id);

    await ensureAgentLoaded(archived.id, { agentManager: manager, agentStorage: storage, logger });
    await ensureAgentLoaded(active.id, { agentManager: manager, agentStorage: storage, logger });

    expect(resumeOptions).toEqual([{ purpose: "history" }, { purpose: "interactive" }]);
  } finally {
    await Promise.all([
      manager.closeAgent(archivedId).catch(() => undefined),
      manager.closeAgent(activeId).catch(() => undefined),
    ]);
    await manager.flush().catch(() => undefined);
    await storage.flush().catch(() => undefined);
    await rm(root, { recursive: true, force: true });
  }
});

test("resuming a stored agent keeps its unread flag and its last-activity time", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "agent-loading-resume-"));
  const logger = createTestLogger();
  const storage = new AgentStorage(path.join(root, "agents"), logger);
  const manager = new AgentManager({
    clients: createTestAgentClients(),
    registry: storage,
    logger,
  });

  const agentId = "00000000-0000-4000-8000-000000000401";
  const lastActive = "2026-01-02T03:04:05.000Z";
  const markedUnread = "2026-01-09T03:04:05.000Z";

  try {
    const agent = await manager.createAgent({ provider: "codex", cwd: root }, agentId, {
      workspaceId: "workspace-a",
    });
    await manager.closeAgent(agent.id);
    await manager.flush();
    await storage.flush();

    const stored = await storage.get(agentId);
    if (!stored) {
      throw new Error("expected a stored agent");
    }
    // The agent finished days ago, and was marked unread later without being opened, which
    // moves `updatedAt` on its own. Clients already hold that newer time, and
    // `acceptAgentDirectoryUpdate` drops anything older, so the resumed agent must not come
    // back carrying only `lastActivityAt`.
    await storage.upsert({
      ...stored,
      updatedAt: markedUnread,
      lastActivityAt: lastActive,
      requiresAttention: true,
      attentionReason: "finished",
      attentionTimestamp: lastActive,
    });

    await ensureAgentLoaded(agentId, { agentManager: manager, agentStorage: storage, logger });
    await manager.flush();
    await storage.flush();

    // Loading the runtime is neither the agent working nor the user reading the chat.
    // Forging either rewrites the workspace's "last used" and drops it out of Ready to review.
    const resumed = await storage.get(agentId);
    expect(resumed?.requiresAttention).toBe(true);
    expect(resumed?.attentionReason).toBe("finished");
    expect(resumed?.updatedAt).toBe(markedUnread);
    expect(resumed?.lastActivityAt).toBe(markedUnread);
    expect(manager.getAgent(agentId)?.attention.requiresAttention).toBe(true);
    expect(manager.getAgent(agentId)?.updatedAt.toISOString()).toBe(markedUnread);
  } finally {
    await manager.closeAgent(agentId).catch(() => undefined);
    await manager.flush().catch(() => undefined);
    await storage.flush().catch(() => undefined);
    await rm(root, { recursive: true, force: true });
  }
});

test("loads an archived agent's history after its working directory is removed", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "agent-loading-missing-cwd-"));
  const worktree = path.join(root, "managed-worktree");
  await mkdir(worktree, { recursive: true });
  const logger = createTestLogger();
  const storage = new AgentStorage(path.join(root, "agents"), logger);
  const manager = new AgentManager({
    clients: createTestAgentClients(),
    registry: storage,
    logger,
  });

  const agentId = "00000000-0000-4000-8000-000000000501";

  try {
    const agent = await manager.createAgent({ provider: "codex", cwd: worktree }, agentId, {
      workspaceId: "workspace-worktree",
    });
    await startAgentRun(manager, agent.id, "what did you change", logger, {});
    // Dispatching a run does not finish it: the provider appends the reply to its
    // history afterwards, and the turn is finalized only once that append lands.
    // Archive after the turn is finalized so the transcript this test reads back is
    // already on disk when the worktree goes away.
    const finished = await manager.waitForAgentEvent(agent.id);
    expect(finished.status).toBe("idle");
    await manager.archiveAgent(agent.id);
    await manager.closeAgent(agent.id);
    await manager.flush();
    await storage.flush();

    // Archiving the workspace removes the worktree it owned. The agent's history is
    // persisted and reading it must not depend on that directory still being there.
    await rm(worktree, { recursive: true, force: true });

    const loaded = await ensureAgentLoaded(agentId, {
      agentManager: manager,
      agentStorage: storage,
      logger,
    });

    expect(loaded.id).toBe(agentId);
    // The transcript is replayed from the provider's persisted history, so the reply the
    // agent gave before the worktree went away is still readable.
    const replies = manager
      .getTimeline(agentId)
      .filter((item) => item.type === "assistant_message");
    expect(replies.length).toBeGreaterThan(0);
    expect(replies.every((item) => item.type === "assistant_message" && item.text.length > 0)).toBe(
      true,
    );
  } finally {
    await manager.closeAgent(agentId).catch(() => undefined);
    await manager.flush().catch(() => undefined);
    await storage.flush().catch(() => undefined);
    await rm(root, { recursive: true, force: true });
  }
});

// A client whose provider replays history without a runtime (as OMP does), recording every
// resume so the test sees whether a read would have started the provider.
function createHistoryReplayClient(resumePurposes: Array<string | undefined>): AgentClient {
  const baseClient = createTestAgentClients().codex;
  if (!baseClient) {
    throw new Error("expected Codex test client");
  }
  return {
    provider: baseClient.provider,
    capabilities: { ...baseClient.capabilities, supportsRuntimeFreeHistory: true },
    createSession: async (config, launchContext) =>
      await baseClient.createSession(config, launchContext),
    resumeSession: async (handle, overrides, launchContext, options) => {
      resumePurposes.push(options?.purpose);
      return await baseClient.resumeSession(handle, overrides, launchContext);
    },
    fetchCatalog: async (options) => await baseClient.fetchCatalog(options),
    isAvailable: async () => await baseClient.isAvailable(),
  };
}

test("reading a closed agent never starts it, and waking it keeps the timeline it served", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "agent-loading-sleep-read-"));
  const logger = createTestLogger();
  const storage = new AgentStorage(path.join(root, "agents"), logger);
  const resumePurposes: Array<string | undefined> = [];
  const managerOptions = () => ({
    clients: { codex: createHistoryReplayClient(resumePurposes) },
    registry: storage,
    logger,
  });
  let manager = new AgentManager(managerOptions());
  const agentId = "00000000-0000-4000-8000-000000000601";

  try {
    const agent = await manager.createAgent({ provider: "codex", cwd: root }, agentId, {
      workspaceId: "workspace-sleep",
    });
    await startAgentRun(manager, agent.id, "remember the codeword", logger, {});
    expect((await manager.waitForAgentEvent(agent.id)).status).toBe("idle");
    const before = manager.fetchTimeline(agentId, { direction: "tail", limit: 0 });
    await manager.closeAgent(agentId);

    // Closed in this daemon run: the retained timeline is served as is, with no resume.
    const deps = { agentManager: manager, agentStorage: storage, logger };
    expect((await ensureAgentHistoryLoaded(agentId, deps)).live).toBe(false);
    const retained = manager.fetchTimeline(agentId, { direction: "tail", limit: 0 });
    expect(retained.epoch).toBe(before.epoch);
    expect(retained.rows).toEqual(before.rows);
    expect(resumePurposes).toEqual([]);

    // After a daemon restart nothing is retained: the history is replayed without a runtime.
    await manager.flush();
    await storage.flush();
    manager = new AgentManager(managerOptions());
    const restartedDeps = { agentManager: manager, agentStorage: storage, logger };
    const loaded = await ensureAgentHistoryLoaded(agentId, restartedDeps);
    expect(loaded.live).toBe(false);
    expect(manager.getAgent(agentId)).toBeNull();
    const replayed = manager.fetchTimeline(agentId, { direction: "tail", limit: 0 });
    expect(replayed.rows.some((row) => row.item.type === "assistant_message")).toBe(true);
    expect(resumePurposes).toEqual(["history"]);

    // Focusing the tab clears the finished-turn attention in storage only.
    expect((await storage.get(agentId))?.requiresAttention).toBe(true);
    await manager.clearAgentAttention(agentId);
    expect((await storage.get(agentId))?.requiresAttention).toBe(false);
    expect(resumePurposes).toEqual(["history"]);

    // A real action wakes it. The timeline the client already holds stays valid: same
    // epoch, same rows, nothing replayed a second time.
    const woken = await ensureAgentLoaded(agentId, restartedDeps);
    expect(woken.lifecycle).toBe("idle");
    expect(resumePurposes).toEqual(["history", "interactive"]);
    const afterWake = manager.fetchTimeline(agentId, { direction: "tail", limit: 0 });
    expect(afterWake.epoch).toBe(replayed.epoch);
    expect(afterWake.rows).toEqual(replayed.rows);
  } finally {
    await manager.closeAgent(agentId).catch(() => undefined);
    await manager.flush().catch(() => undefined);
    await storage.flush().catch(() => undefined);
    await rm(root, { recursive: true, force: true });
  }
});
