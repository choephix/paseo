import { readdir, readFile } from "node:fs/promises";
import type { Logger } from "pino";

import { withTimeout } from "../../utils/promise-timeout.js";
import { ensureUnarchivedAgentLoaded } from "./agent-loading.js";
import type { AgentManager, ManagedAgent } from "./agent-manager.js";
import type { AgentSlashCommand } from "./agent-sdk-types.js";
import type { AgentStorage } from "./agent-storage.js";

// Sleeping agents. Sleeping closes an idle agent's runtime to free its memory; the agent
// stays registered, its conversation stays readable without starting it (see
// ensureAgentHistoryLoaded), and sending to it or waking it starts it again. Only providers
// that replay history without a runtime (supportsRuntimeFreeHistory, today OMP) can sleep:
// any other agent would just be started again by the next read.

export interface AgentSleepMarker {
  sleptAt: string;
  reason: "idle" | "manual";
  /** Resident memory of the runtime's process tree when it was put to sleep. */
  rssBytes: number | null;
}

/** Why an agent cannot be put to sleep right now. */
export type AgentAwakeReason = "running" | "permission" | "background_work";

export interface AsleepAgent {
  agentId: string;
  provider: string;
  /** Null for a closed agent the daemon did not put to sleep itself (e.g. after a restart). */
  sleep: AgentSleepMarker | null;
}

export interface AgentSleepServiceOptions {
  agentManager: AgentManager;
  agentStorage: AgentStorage;
  logger: Logger;
}

const BACKGROUND_WORK_PROBE_TIMEOUT_MS = 5_000;

export class AgentSleepService {
  private readonly agentManager: AgentManager;
  private readonly agentStorage: AgentStorage;
  private readonly logger: Logger;
  // Slash commands listed while the runtime was up, so typing `/` in a sleeping agent's
  // composer still offers them without starting it.
  private readonly commandsAtSleep = new Map<string, AgentSlashCommand[]>();

  constructor(options: AgentSleepServiceOptions) {
    this.agentManager = options.agentManager;
    this.agentStorage = options.agentStorage;
    this.logger = options.logger.child({ module: "agent-sleep" });
  }

  /**
   * Closes a live, idle agent to free its runtime. Returns the stored marker, or the existing
   * one (possibly null) when the agent is already closed.
   */
  async sleepAgent(
    agentId: string,
    reason: AgentSleepMarker["reason"],
  ): Promise<AgentSleepMarker | null> {
    const agent = this.agentManager.getAgent(agentId);
    if (!agent) {
      const record = await this.agentStorage.get(agentId);
      if (!record || record.archivedAt || record.internal) {
        throw new Error(`Agent not found: ${agentId}`);
      }
      return record.sleep ?? null;
    }
    if (agent.internal) {
      throw new Error(`Agent not found: ${agentId}`);
    }
    if (!this.agentManager.supportsRuntimeFreeHistory(agent.provider)) {
      throw new Error(
        `Agents of provider '${agent.provider}' cannot sleep: reading their conversation would start them again`,
      );
    }
    const awake = await this.findAwakeReason(agent);
    if (awake) {
      throw new Error(`Agent cannot sleep now: ${awake}`);
    }

    const rssBytes = await readProcessTreeRssBytes(agent.session?.getProcessId?.() ?? null);
    const commands = await agent.session?.listCommands?.().catch(() => null);
    // A turn may have started while the runtime was measured.
    const current = this.agentManager.getAgent(agentId);
    if (!current || this.findBusyReason(current)) {
      throw new Error("Agent cannot sleep now: running");
    }

    await this.agentManager.closeAgent(agentId);
    if (commands) {
      this.commandsAtSleep.set(agentId, commands);
    }
    const marker: AgentSleepMarker = { sleptAt: new Date().toISOString(), reason, rssBytes };
    const record = await this.agentStorage.get(agentId);
    if (record) {
      await this.agentStorage.upsert({ ...record, sleep: marker });
    }
    this.logger.info({ agentId, reason, rssBytes }, "Agent put to sleep");
    return marker;
  }

  /** Starts a closed agent again. A live agent is returned as is. */
  async wakeAgent(agentId: string): Promise<ManagedAgent> {
    const live = this.agentManager.getAgent(agentId);
    if (live) {
      return live;
    }
    const agent = await ensureUnarchivedAgentLoaded(agentId, {
      agentManager: this.agentManager,
      agentStorage: this.agentStorage,
      logger: this.logger,
      broadcastTimeline: true,
    });
    this.logger.info({ agentId }, "Agent woken");
    return agent;
  }

  /** Closed, unarchived agents whose conversation is served without starting them. */
  async listAsleepAgents(): Promise<AsleepAgent[]> {
    const records = await this.agentStorage.list();
    return records
      .filter(
        (record) =>
          !record.archivedAt &&
          !record.internal &&
          !this.agentManager.getAgent(record.id) &&
          this.agentManager.supportsRuntimeFreeHistory(record.provider),
      )
      .map((record) => ({
        agentId: record.id,
        provider: record.provider,
        sleep: record.sleep ?? null,
      }));
  }

  getCommandsAtSleep(agentId: string): AgentSlashCommand[] | null {
    return this.commandsAtSleep.get(agentId) ?? null;
  }

  private async findAwakeReason(agent: ManagedAgent): Promise<AgentAwakeReason | null> {
    const busy = this.findBusyReason(agent);
    if (busy) {
      return busy;
    }
    const subagents = this.agentManager.listProviderSubagents(agent.id);
    if (subagents.some((subagent) => subagent.status === "running")) {
      return "background_work";
    }
    const probe = agent.session?.hasPendingBackgroundWork?.();
    if (!probe) {
      return null;
    }
    try {
      const pending = await withTimeout(
        probe,
        BACKGROUND_WORK_PROBE_TIMEOUT_MS,
        "Timed out asking the agent runtime about background work",
      );
      return pending ? "background_work" : null;
    } catch (error) {
      // Unknown is treated as busy: sleeping would kill whatever is running.
      this.logger.warn({ err: error, agentId: agent.id }, "Background work probe failed");
      return "background_work";
    }
  }

  private findBusyReason(agent: ManagedAgent): AgentAwakeReason | null {
    if (agent.lifecycle !== "idle" || this.agentManager.hasInFlightRun(agent.id)) {
      return "running";
    }
    if (agent.pendingPermissions.size > 0) {
      return "permission";
    }
    return null;
  }
}

// Sum of VmRSS over a process and its descendants (Linux /proc); null elsewhere or if gone.
async function readProcessTreeRssBytes(rootPid: number | null): Promise<number | null> {
  if (rootPid === null || process.platform !== "linux") {
    return null;
  }
  const seen = new Set<number>();
  const pending = [rootPid];
  let total = 0;
  while (pending.length > 0) {
    const pid = pending.pop()!;
    if (seen.has(pid)) {
      continue;
    }
    seen.add(pid);
    const status = await readFile(`/proc/${pid}/status`, "utf8").catch(() => null);
    const rssKb = status ? /^VmRSS:\s+(\d+)\s+kB/m.exec(status)?.[1] : undefined;
    if (!rssKb) {
      continue;
    }
    total += Number(rssKb) * 1024;
    const tasks = await readdir(`/proc/${pid}/task`).catch(() => [] as string[]);
    for (const task of tasks) {
      const children = await readFile(`/proc/${pid}/task/${task}/children`, "utf8").catch(() => "");
      for (const child of children.split(/\s+/)) {
        if (child) pending.push(Number(child));
      }
    }
  }
  return total > 0 ? total : null;
}
