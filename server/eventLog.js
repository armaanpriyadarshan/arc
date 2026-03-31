/**
 * AgentWall Event Log
 *
 * Records every tool call decision (allowed / blocked / redacted) for the
 * activity feed and provides real-time broadcast over WebSocket.
 */

import WebSocket from "ws";

/**
 * In-memory event store with auto-incrementing IDs and filtering.
 */
export class EventLog {
  constructor() {
    this._events = [];
    this._counter = 0;
  }

  /**
   * Record a new event. Auto-generates `id` and `timestamp`.
   *
   * @param {object} eventData - Partial event (everything except id and timestamp)
   * @returns {object} The fully-formed event that was stored
   */
  add(eventData) {
    this._counter++;
    const event = {
      id: `evt_${String(this._counter).padStart(3, "0")}`,
      timestamp: new Date().toISOString(),
      ...eventData
    };
    this._events.push(event);
    return event;
  }

  /**
   * Retrieve events in reverse-chronological order with optional filters.
   *
   * @param {object} [filters]
   * @param {string} [filters.status]         - Filter by "allowed" | "blocked" | "redacted"
   * @param {string} [filters.source_system]  - Filter by source system name
   * @param {number} [filters.limit]          - Maximum number of events to return
   * @param {number} [filters.offset]         - Number of events to skip before returning
   * @returns {Array} Filtered, reverse-chronologically ordered events
   */
  getAll(filters) {
    let result = [...this._events].reverse();

    if (filters) {
      if (filters.status) {
        result = result.filter((e) => e.status === filters.status);
      }
      if (filters.source_system) {
        result = result.filter((e) => e.source_system === filters.source_system);
      }
      if (typeof filters.offset === "number" && filters.offset > 0) {
        result = result.slice(filters.offset);
      }
      if (typeof filters.limit === "number" && filters.limit > 0) {
        result = result.slice(0, filters.limit);
      }
    }

    return result;
  }

  /** Clear all events and reset the counter. */
  clear() {
    this._events = [];
    this._counter = 0;
  }
}

/**
 * Broadcast an event to every connected WebSocket client.
 *
 * @param {WebSocketServer} wss  - The ws WebSocketServer instance
 * @param {object}          event - The event object to broadcast
 */
export function broadcastEvent(wss, event) {
  if (!wss || !wss.clients) return;

  const message = JSON.stringify({ type: "tool_call", event });

  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  }
}
