import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { Logger } from "pino";
import { afterEach, expect, test } from "vitest";

import { createTestLogger } from "../../test-utils/test-logger.js";
import { createTestAgentClients } from "../test-utils/fake-agent-client.js";
import { ensureAgentHistoryLoaded } from "./agent-loading.js";
import { AgentManager } from "./agent-manager.js";
import { sendPromptToAgent, startAgentRun } from "./agent-prompt.js";
import type { AgentClient, AgentSession } from "./agent-sdk-types.js";
import { AgentSleepService } from "./agent-sleep.js";
import { AgentStorage } from "./agent-storage.js";

interface Harness {
  manager: AgentManager;
  storage: AgentStorage;
  sleep: AgentSleepService;
  resumePurposes: Array<string | undefined>;
  background: { pending: boolean };
  logger: Logger;
  root: string;
  clock: { now: number };
  open: Set<string>;
}

const harnesses: Harness[] = [];

afterEach(async () => {
  for (const harness of harnesses.splice(0)) {
    for (const agent of harness.manager.listAgents()) {
      await harness.manager.closeAgent(agent.id).catch(() => undefined);
    }
    await harness.manager.flush().catch(() => undefined);
    await harness.storage.flush().catch(() => undefined);
    await rm(harness.root, { recursive: true, force: true });
  }
});

// A provider that replays history without a runtime (as OMP does). Every resume is recorded,
// so a test sees whether something would have started the provider; sessions report
// background work from `background.pending`.
async function createHarness(): Promise<Harness> {
  const root = await mkdtemp(path.join(tmpdir(), "agent-sleep-"));
  const logger = createTestLogger();
  const storage = new AgentStorage(path.join(root, "agents"), logger);
  const resumePurposes: Array<string | undefined> = [];
  const background = { pending: false };
  const baseClient = createTestAgentClients().codex;
  if (!baseClient) {
    throw new Error("expected Codex test client");
  }
  const withBackgroundProbe = (session: AgentSession): AgentSession =>
    Object.assign(session, { hasPendingBackgroundWork: async () => background.pending });
  const client: AgentClient = {
    provider: baseClient.provider,
    capabilities: { ...baseClient.capabilities, supportsRuntimeFreeHistory: true },
    createSession: async (config, launchContext) =>
      withBackgroundProbe(await baseClient.createSession(config, launchContext)),
    resumeSession: async (handle, overrides, launchContext, options) => {
      resumePurposes.push(options?.purpose);
      return withBackgroundProbe(await baseClient.resumeSession(handle, overrides, launchContext));
    },
    fetchCatalog: async (options) => await baseClient.fetchCatalog(options),
    isAvailable: async () => await baseClient.isAvailable(),
  };
  const manager = new AgentManager({ clients: { codex: client }, registry: storage, logger });
  const clock = { now: Date.now() };
  const open = new Set<string>();
  const sleep = new AgentSleepService({
    agentManager: manager,
    agentStorage: storage,
    logger,
    getSleepConfig: () => ({ enabled: true, idleMinutes: 1 }),
    isAgentOpenInClient: (agentId) => open.has(agentId),
    now: () => clock.now,
  });
  const harness = {
    manager,
    storage,
    sleep,
    resumePurposes,
    background,
    logger,
    root,
    clock,
    open,
  };
  harnesses.push(harness);
  return harness;
}

async function createIdleAgent(harness: Harness, agentId: string, prompt: string) {
  const agent = await harness.manager.createAgent(
    { provider: "codex", cwd: harness.root },
    agentId,
    { workspaceId: "workspace-sleep" },
  );
  await startAgentRun(harness.manager, agent.id, prompt, harness.logger, {});
  expect((await harness.manager.waitForAgentEvent(agent.id)).status).toBe("idle");
  return agent;
}

test("a slept agent stays closed while read, and a send wakes it with its memory", async () => {
  const harness = await createHarness();
  const { manager, storage, sleep, resumePurposes, logger } = harness;
  const agentId = "00000000-0000-4000-8000-000000000701";
  await createIdleAgent(harness, agentId, 'Please remember this marker "BLUE-HERON".');

  const marker = await sleep.sleepAgent(agentId, "manual");
  expect(marker?.reason).toBe("manual");
  expect(manager.getAgent(agentId)).toBeNull();
  expect((await storage.get(agentId))?.sleep).toEqual(marker);
  expect(await sleep.listAsleepAgents()).toEqual([{ agentId, provider: "codex", sleep: marker }]);

  // Opening its tab reads the conversation and leaves it asleep.
  const read = await ensureAgentHistoryLoaded(agentId, {
    agentManager: manager,
    agentStorage: storage,
    logger,
  });
  expect(read.live).toBe(false);
  expect(resumePurposes).toEqual([]);

  // Sending wakes it, and it remembers what it was told before sleeping.
  await sendPromptToAgent({
    agentManager: manager,
    agentStorage: storage,
    agentId,
    prompt: "What was the marker?",
    logger,
  });
  expect((await manager.waitForAgentEvent(agentId)).status).toBe("idle");
  expect(resumePurposes).toEqual(["interactive"]);
  // The fake's answer to "what was the marker" is the marker it was told before sleeping.
  expect(await manager.getLastAssistantMessage(agentId)).toMatch(/BLUE-HERON$/);
  await manager.flush();
  await storage.flush();
  expect((await storage.get(agentId))?.sleep ?? null).toBeNull();
  expect(await sleep.listAsleepAgents()).toEqual([]);
});

test("an agent with background work is not put to sleep", async () => {
  const harness = await createHarness();
  const agentId = "00000000-0000-4000-8000-000000000702";
  await createIdleAgent(harness, agentId, "Say hello.");

  harness.background.pending = true;
  await expect(harness.sleep.sleepAgent(agentId, "manual")).rejects.toThrow("background_work");
  expect(harness.manager.getAgent(agentId)?.lifecycle).toBe("idle");

  harness.background.pending = false;
  await harness.sleep.sleepAgent(agentId, "manual");
  expect(harness.manager.getAgent(agentId)).toBeNull();

  // Waking it explicitly starts it again.
  const woken = await harness.sleep.wakeAgent(agentId);
  expect(woken.lifecycle).toBe("idle");
  expect(harness.resumePurposes).toEqual(["interactive"]);
});

test("the idle timer sleeps an idle agent only once nothing keeps it awake", async () => {
  const harness = await createHarness();
  const { manager, sleep, clock, open, background } = harness;
  const agentId = "00000000-0000-4000-8000-000000000703";
  await createIdleAgent(harness, agentId, "Say hello.");

  // Not idle for a minute yet.
  await sleep.sweep();
  expect(manager.getAgent(agentId)).not.toBeNull();

  // Idle long enough while its tab stays open on a device, sweep after sweep.
  open.add(agentId);
  for (let sweep = 0; sweep < 5; sweep += 1) {
    clock.now += 30_000;
    await harness.sleep.sweep();
  }
  expect(manager.getAgent(agentId)).not.toBeNull();
  expect(sleep.listKeptAwake()).toMatchObject([{ agentId, reason: "open_in_client" }]);

  // Closing the tab restarts the idle minute; then background work keeps it awake.
  open.delete(agentId);
  await sleep.sweep();
  expect(manager.getAgent(agentId)).not.toBeNull();
  clock.now += 2 * 60_000;
  background.pending = true;
  await sleep.sweep();
  expect(manager.getAgent(agentId)).not.toBeNull();
  expect(sleep.listKeptAwake()).toMatchObject([{ agentId, reason: "background_work" }]);

  background.pending = false;
  await sleep.sweep();
  expect(manager.getAgent(agentId)).toBeNull();
  expect((await harness.storage.get(agentId))?.sleep?.reason).toBe("idle");
  expect(sleep.listKeptAwake()).toEqual([]);
});

test("a prompt on its way to an agent is never cut off by sleep", async () => {
  const harness = await createHarness();
  const { manager, storage, sleep, resumePurposes, logger } = harness;
  const agentId = "00000000-0000-4000-8000-000000000704";
  await createIdleAgent(harness, agentId, 'Please remember this marker "RED-KITE".');

  // Hold the send between loading the agent and starting its turn: there the agent still
  // looks idle to a sleep check.
  const session = manager.getAgent(agentId)?.session;
  if (!session) throw new Error("expected a live session");
  let reachedMode!: () => void;
  const atMode = new Promise<void>((resolve) => (reachedMode = resolve));
  let releaseMode!: () => void;
  const modeReleased = new Promise<void>((resolve) => (releaseMode = resolve));
  session.setMode = async () => {
    reachedMode();
    await modeReleased;
  };
  const send = sendPromptToAgent({
    agentManager: manager,
    agentStorage: storage,
    agentId,
    prompt: "What was the marker?",
    sessionMode: "default",
    logger,
  });
  await atMode;

  await expect(sleep.sleepAgent(agentId, "manual")).rejects.toThrow("running");
  releaseMode();
  await send;
  expect((await manager.waitForAgentEvent(agentId)).status).toBe("idle");
  expect(await manager.getLastAssistantMessage(agentId)).toMatch(/RED-KITE$/);
  expect(resumePurposes).toEqual([]);
  expect((await storage.get(agentId))?.sleep ?? null).toBeNull();
});
