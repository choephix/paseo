import { readdir, readFile } from "node:fs/promises";
import type { Logger } from "pino";

import { withTimeout } from "../../utils/promise-timeout.js";
import { ensureUnarchivedAgentLoaded } from "./agent-loading.js";
import type { AgentManager, ManagedAgent } from "./agent-manager.js";
import type { AgentSlashCommand } from "./agent-sdk-types.js";
import type { AgentStorage } from "./agent-storage.js";
import type { ProviderSleepConfig } from "@getpaseo/protocol/provider-config";

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

/** Why the idle timer left an idle-long-enough agent awake. */
export type AgentKeptAwakeReason = AgentAwakeReason | "open_in_client";

export interface AgentKeptAwake {
  agentId: string;
  reason: AgentKeptAwakeReason;
  /** When the timer first kept it awake for this reason. */
  since: string;
}

export interface AgentSleepConfig {
  provider: string;
  enabled: boolean;
  idleMinutes: number;
}

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
  /** The provider's `agents.providers.<id>.sleep` config, if any. Read on every sweep. */
  getSleepConfig?: (provider: string) => ProviderSleepConfig | undefined;
  /** Whether a connected client shows this agent right now (focused in a visible app). */
  isAgentOpenInClient?: (agentId: string) => boolean;
  now?: () => number;
}

export const DEFAULT_SLEEP_IDLE_MINUTES = 60;
const SWEEP_INTERVAL_MS = 30_000;

const BACKGROUND_WORK_PROBE_TIMEOUT_MS = 5_000;

export class AgentSleepService {
  private readonly agentManager: AgentManager;
  private readonly agentStorage: AgentStorage;
  private readonly logger: Logger;
  private readonly getSleepConfig: NonNullable<AgentSleepServiceOptions["getSleepConfig"]>;
  private readonly isAgentOpenInClient: NonNullable<
    AgentSleepServiceOptions["isAgentOpenInClient"]
  >;
  private readonly now: () => number;
  // Idle time counts from the last of: activity (updatedAt), the first sweep that saw the agent
  // live (a woken agent keeps its old updatedAt), and the last sweep that saw it open.
  private readonly liveSince = new Map<string, number>();
  private readonly lastOpenAt = new Map<string, number>();
  private readonly keptAwake = new Map<string, AgentKeptAwake>();
  private sweepTimer: ReturnType<typeof setInterval> | null = null;
  private sweeping = false;
  // Slash commands listed while the runtime was up, so typing `/` in a sleeping agent's
  // composer still offers them without starting it.
  private readonly commandsAtSleep = new Map<string, AgentSlashCommand[]>();

  constructor(options: AgentSleepServiceOptions) {
    this.agentManager = options.agentManager;
    this.agentStorage = options.agentStorage;
    this.logger = options.logger.child({ module: "agent-sleep" });
    this.getSleepConfig = options.getSleepConfig ?? (() => undefined);
    this.isAgentOpenInClient = options.isAgentOpenInClient ?? (() => false);
    this.now = options.now ?? Date.now;
  }

  start(): void {
    if (this.sweepTimer) return;
    this.sweepTimer = setInterval(() => {
      void this.sweep().catch((error: unknown) => {
        this.logger.warn({ err: error }, "Idle sleep sweep failed");
      });
    }, SWEEP_INTERVAL_MS);
    this.sweepTimer.unref?.();
  }

  stop(): void {
    if (this.sweepTimer) clearInterval(this.sweepTimer);
    this.sweepTimer = null;
  }

  /** Sleep config of every provider whose agents can sleep. */
  listSleepConfigs(): AgentSleepConfig[] {
    return this.agentManager
      .getRegisteredProviderIds()
      .filter((provider) => this.agentManager.supportsRuntimeFreeHistory(provider))
      .map((provider) => this.resolveSleepConfig(provider));
  }

  listKeptAwake(): AgentKeptAwake[] {
    return [...this.keptAwake.values()];
  }

  /**
   * Puts to sleep every live agent whose provider has sleep enabled and that has been idle
   * for its provider's idleMinutes, unless it is running, waiting on a permission, has
   * background work, or is open in a client. Skips are recorded for listKeptAwake.
   */
  async sweep(): Promise<void> {
    if (this.sweeping) return;
    this.sweeping = true;
    try {
      const now = this.now();
      const live = this.agentManager.listAgents();
      const liveIds = new Set(live.map((agent) => agent.id));
      for (const agentId of this.liveSince.keys()) {
        if (liveIds.has(agentId)) continue;
        this.liveSince.delete(agentId);
        this.lastOpenAt.delete(agentId);
        this.keptAwake.delete(agentId);
      }
      for (const agent of live) {
        await this.considerIdleAgent(agent, now);
      }
    } finally {
      this.sweeping = false;
    }
  }

  private async considerIdleAgent(agent: ManagedAgent, now: number): Promise<void> {
    const liveSince = this.liveSince.get(agent.id) ?? now;
    this.liveSince.set(agent.id, liveSince);
    const open = this.isAgentOpenInClient(agent.id);
    const lastActiveAt = Math.max(
      agent.updatedAt.getTime(),
      liveSince,
      this.lastOpenAt.get(agent.id) ?? 0,
    );
    if (open) this.lastOpenAt.set(agent.id, now);

    const config = this.resolveSleepConfig(agent.provider);
    const candidate = agent.lifecycle === "idle" || agent.lifecycle === "running";
    if (
      !config.enabled ||
      !candidate ||
      !this.agentManager.supportsRuntimeFreeHistory(agent.provider) ||
      now - lastActiveAt < config.idleMinutes * 60_000
    ) {
      this.keptAwake.delete(agent.id);
      return;
    }

    const reason: AgentKeptAwakeReason | null =
      this.findBusyReason(agent) ?? (open ? "open_in_client" : await this.findAwakeReason(agent));
    if (reason) {
      if (this.keptAwake.get(agent.id)?.reason !== reason) {
        this.keptAwake.set(agent.id, {
          agentId: agent.id,
          reason,
          since: new Date(now).toISOString(),
        });
      }
      return;
    }
    this.keptAwake.delete(agent.id);
    await this.sleepAgent(agent.id, "idle").catch((error: unknown) => {
      this.logger.warn({ err: error, agentId: agent.id }, "Failed to put idle agent to sleep");
    });
  }

  private resolveSleepConfig(provider: string): AgentSleepConfig {
    const config = this.getSleepConfig(provider);
    return {
      provider,
      enabled: config?.enabled === true,
      idleMinutes: config?.idleMinutes ?? DEFAULT_SLEEP_IDLE_MINUTES,
    };
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
