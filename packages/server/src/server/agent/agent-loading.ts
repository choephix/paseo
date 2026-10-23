import type { Logger } from "pino";

import type { AgentProvider } from "./agent-sdk-types.js";
import type { AgentManager, ManagedAgent } from "./agent-manager.js";
import type { AgentStorage, StoredAgentRecord } from "./agent-storage.js";
import {
  buildConfigOverrides,
  buildSessionConfig,
  extractAttention,
  extractTimestamps,
  isStoredAgentProviderAvailable,
  toAgentPersistenceHandle,
} from "../persistence-hooks.js";

export class AgentNotFoundError extends Error {
  constructor(agentId: string) {
    super(`Agent not found: ${agentId}`);
    this.name = "AgentNotFoundError";
  }
}

interface PendingAgentInitialization {
  promise: Promise<ManagedAgent>;
  options: { broadcastTimeline: boolean };
}

const pendingAgentInitializations = new Map<string, PendingAgentInitialization>();

export type AgentLoaderManager = Pick<
  AgentManager,
  | "createAgent"
  | "getAgent"
  | "getRegisteredProviderIds"
  | "hydrateTimelineFromProvider"
  | "resumeAgentFromPersistence"
> &
  Partial<Pick<AgentManager, "waitForAgentClose">>;

export interface EnsureAgentLoadedDeps {
  agentManager: AgentLoaderManager;
  agentStorage: AgentStorage;
  validProviders?: Iterable<AgentProvider>;
  broadcastTimeline?: boolean;
  logger: Logger;
}

export async function ensureUnarchivedAgentLoaded(
  agentId: string,
  deps: EnsureAgentLoadedDeps & {
    agentManager: AgentLoaderManager & Pick<AgentManager, "closeAgent">;
  },
): Promise<ManagedAgent> {
  const record = await deps.agentStorage.get(agentId);
  if (record?.archivedAt) {
    throw new Error(`Agent is archived: ${agentId}`);
  }

  const agent = await ensureAgentLoaded(agentId, deps);
  const latestRecord = await deps.agentStorage.get(agentId);
  if (latestRecord?.archivedAt) {
    await deps.agentManager.closeAgent(agentId).catch((error: unknown) => {
      deps.logger.warn({ err: error, agentId }, "Failed to close concurrently archived agent");
    });
    throw new Error(`Agent is archived: ${agentId}`);
  }

  return agent;
}

export async function ensureAgentLoaded(
  agentId: string,
  deps: EnsureAgentLoadedDeps,
): Promise<ManagedAgent> {
  await deps.agentManager.waitForAgentClose?.(agentId);

  const inflight = pendingAgentInitializations.get(agentId);
  if (inflight) {
    inflight.options.broadcastTimeline ||= deps.broadcastTimeline === true;
    return inflight.promise;
  }

  const existing = deps.agentManager.getAgent(agentId);
  if (existing) {
    return existing;
  }

  // A close may have started after the first barrier observed no in-flight
  // work. Once the live lookup is empty, this second barrier closes that gap
  // before storage-backed resume begins.
  await deps.agentManager.waitForAgentClose?.(agentId);

  const laterInflight = pendingAgentInitializations.get(agentId);
  if (laterInflight) {
    laterInflight.options.broadcastTimeline ||= deps.broadcastTimeline === true;
    return laterInflight.promise;
  }

  const pendingOptions = {
    broadcastTimeline: deps.broadcastTimeline === true,
  };
  const initPromise = (async () => {
    const record = await deps.agentStorage.get(agentId);
    if (!record) {
      throw new AgentNotFoundError(agentId);
    }

    const validProviders = deps.validProviders ?? deps.agentManager.getRegisteredProviderIds();
    if (!isStoredAgentProviderAvailable(record, validProviders)) {
      throw new Error(`Agent ${agentId} references unavailable provider '${record.provider}'`);
    }

    const handle = toAgentPersistenceHandle(validProviders, record.persistence);

    let snapshot: ManagedAgent;
    if (handle) {
      snapshot = await deps.agentManager.resumeAgentFromPersistence(
        handle,
        buildConfigOverrides(record),
        agentId,
        { ...extractTimestamps(record), attention: extractAttention(record) },
        record.archivedAt ? { purpose: "history" } : undefined,
      );
      deps.logger.info({ agentId, provider: record.provider }, "Agent resumed from persistence");
    } else {
      // No provider handle to resume: this starts the agent's first session rather than
      // bringing one back, so it stamps activity and carries no stored attention. Records
      // without a handle never got far enough to accumulate either.
      const config = buildSessionConfig(record, {
        validProviders,
      });
      if (!config) {
        throw new Error(`Agent ${agentId} references unavailable provider '${record.provider}'`);
      }
      snapshot = await deps.agentManager.createAgent(config, agentId, {
        labels: record.labels,
        workspaceId: record.workspaceId,
        owner: record.owner,
      });
      deps.logger.info({ agentId, provider: record.provider }, "Agent created from stored config");
    }

    await deps.agentManager.hydrateTimelineFromProvider(agentId, {
      broadcast: () => pendingOptions.broadcastTimeline,
    });
    return deps.agentManager.getAgent(agentId) ?? snapshot;
  })();

  const pending: PendingAgentInitialization = { promise: initPromise, options: pendingOptions };
  pendingAgentInitializations.set(agentId, pending);

  try {
    return await initPromise;
  } finally {
    const current = pendingAgentInitializations.get(agentId);
    if (current === pending) {
      pendingAgentInitializations.delete(agentId);
    }
  }
}

export type AgentHistoryLoadResult =
  | { live: true; agent: ManagedAgent }
  | { live: false; record: StoredAgentRecord };

export interface EnsureAgentHistoryLoadedDeps extends EnsureAgentLoadedDeps {
  agentManager: AgentLoaderManager & Pick<AgentManager, "loadTimelineWithoutRuntime">;
  /** Loads agents whose provider cannot replay history without a runtime. */
  loadAgent?: (agentId: string) => Promise<ManagedAgent>;
}

/**
 * Loads an agent for reading its conversation. A live agent is returned as is. A closed,
 * unarchived agent whose provider replays history without a runtime (OMP) stays closed:
 * its timeline is served from that replay, so opening a tab, scrolling, searching or
 * reconnecting never starts it. Only sending, an explicit wake or another real action
 * (those call ensureAgentLoaded) does. Every other agent loads as before.
 */
export async function ensureAgentHistoryLoaded(
  agentId: string,
  deps: EnsureAgentHistoryLoadedDeps,
): Promise<AgentHistoryLoadResult> {
  const loadAgent = deps.loadAgent ?? ((id: string) => ensureAgentLoaded(id, deps));
  await deps.agentManager.waitForAgentClose?.(agentId);
  const inflight = pendingAgentInitializations.get(agentId);
  if (inflight) {
    return { live: true, agent: await inflight.promise };
  }
  const existing = deps.agentManager.getAgent(agentId);
  if (existing) {
    return { live: true, agent: existing };
  }

  const record = await deps.agentStorage.get(agentId);
  if (!record) {
    throw new AgentNotFoundError(agentId);
  }
  const validProviders = deps.validProviders ?? deps.agentManager.getRegisteredProviderIds();
  if (record.archivedAt || !isStoredAgentProviderAvailable(record, validProviders)) {
    return { live: true, agent: await loadAgent(agentId) };
  }
  const loaded = await deps.agentManager.loadTimelineWithoutRuntime(agentId, {
    provider: record.provider,
    handle: toAgentPersistenceHandle(validProviders, record.persistence),
    overrides: buildConfigOverrides(record),
  });
  if (!loaded) {
    return { live: true, agent: await loadAgent(agentId) };
  }
  // A wake may have finished while the history loaded; prefer the live agent then.
  const woken = deps.agentManager.getAgent(agentId);
  return woken ? { live: true, agent: woken } : { live: false, record };
}
