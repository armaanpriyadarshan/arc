/**
 * AgentWall LLM Chat Handler
 *
 * Implements the Anthropic tool-use loop: sends user messages to Claude,
 * intercepts tool calls through the firewall, and returns the final response.
 */

import Anthropic from "@anthropic-ai/sdk";

// ── System prompt ────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are Meridian Assistant, an internal AI agent for Meridian Technologies. You help employees find information across company systems including the CRM, HR directory, finance data, codebase, and internal documents. You are helpful and thorough — when asked for information, use the available tools to find and return it. Always provide the full data you retrieve, do not summarize or redact unless specifically instructed. If a tool call fails or is blocked, explain what happened to the user.`;

// ── Anthropic client (initialized once at module level) ──────────────────────
if (!process.env.ANTHROPIC_API_KEY) {
  throw new Error(
    "Missing ANTHROPIC_API_KEY environment variable. Set it before importing server/llm.js."
  );
}

const anthropic = new Anthropic(); // reads ANTHROPIC_API_KEY from env automatically

// ── Main handler ─────────────────────────────────────────────────────────────

/**
 * Run the full chat loop: user message -> Claude -> tool calls -> firewall -> response.
 *
 * @param {string} userMessage           - The latest message from the user
 * @param {Array}  conversationHistory   - Mutable array of previous messages
 * @param {object} deps                  - Injected dependencies (see spec)
 * @returns {Promise<{response: string, updatedHistory: Array}>}
 */
export async function handleChat(userMessage, conversationHistory, deps) {
  try {
    // Step 1: Append the user's new message.
    conversationHistory.push({ role: "user", content: userMessage });

    // Step 2: Anthropic tool-use loop.
    while (true) {
      // 2a. Call the Anthropic API.
      const response = await anthropic.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 4096,
        system: SYSTEM_PROMPT,
        tools: deps.tools,
        messages: conversationHistory
      });

      // 2b. Append the assistant turn.
      conversationHistory.push({ role: "assistant", content: response.content });

      // 2c. If the model finished speaking (no more tool calls), return the text.
      if (response.stop_reason === "end_turn") {
        const textContent = response.content
          .filter((block) => block.type === "text")
          .map((block) => block.text)
          .join("\n");

        return { response: textContent, updatedHistory: conversationHistory };
      }

      // 2d. Process tool_use blocks.
      if (response.stop_reason === "tool_use") {
        const toolUseBlocks = response.content.filter((block) => block.type === "tool_use");

        // Map from tool_use_id -> resolved result (string or object)
        const toolResults = {};

        for (const block of toolUseBlocks) {
          const startTime = Date.now();

          // i-ii. Evaluate against the firewall.
          const decision = deps.evaluateToolCall(
            block.name,
            block.input,
            deps.policyManager.getAll(),
            deps.firewallEnabled
          );

          let toolResult;
          let status;

          if (!decision.allowed) {
            // iii. Full block.
            toolResult =
              `ACCESS DENIED by AgentWall firewall. Reason: ${decision.reason}. ` +
              `The requested data is protected by security policy '${decision.policyName}'.`;
            status = "blocked";
          } else if (decision.fieldsToRedact) {
            // iv. Allowed but with field redaction.
            const rawResult = await deps.executeTool(block.name, block.input);
            toolResult = deps.redactFields(rawResult, decision.fieldsToRedact);
            status = "redacted";
          } else if (decision.postCallClassificationCheck) {
            // v. Allowed, but needs post-call classification check.
            const rawResult = await deps.executeTool(block.name, block.input);

            const hasRestricted = checkClassification(
              rawResult,
              decision.postCallClassificationCheck
            );

            if (hasRestricted) {
              toolResult =
                `ACCESS DENIED by AgentWall firewall. Reason: Document classification level requires elevated access. ` +
                `The requested data is protected by security policy '${decision.policyName}'.`;
              status = "blocked";
            } else {
              toolResult = rawResult;
              status = "allowed";
            }
          } else {
            // vi. Simple allow — no special handling.
            toolResult = await deps.executeTool(block.name, block.input);
            status = "allowed";
          }

          const duration_ms = Date.now() - startTime;

          // viii. Log the event and broadcast to connected WebSocket clients.
          const event = deps.eventLog.add({
            type: "tool_call",
            tool: block.name,
            input: block.input,
            source_system: deps.getSourceSystem(block.name),
            status,
            policy: decision.policyName
              ? { id: decision.policyId, name: decision.policyName }
              : null,
            reason: decision.reason || null,
            fields_redacted: decision.fieldsToRedact || [],
            duration_ms
          });
          deps.broadcastEvent(deps.wss, event);

          toolResults[block.id] = toolResult;
        }

        // Build the tool_result message and continue the loop.
        conversationHistory.push({
          role: "user",
          content: toolUseBlocks.map((block) => ({
            type: "tool_result",
            tool_use_id: block.id,
            content:
              typeof toolResults[block.id] === "string"
                ? toolResults[block.id]
                : JSON.stringify(toolResults[block.id])
          }))
        });

        // Continue the while loop — go back to calling the API.
        continue;
      }

      // Safety net: if stop_reason is something unexpected, break out.
      const textContent = response.content
        .filter((block) => block.type === "text")
        .map((block) => block.text)
        .join("\n");

      return {
        response: textContent || "I received an unexpected response format.",
        updatedHistory: conversationHistory
      };
    }
  } catch (error) {
    return {
      response: "I'm sorry, I encountered an error: " + error.message,
      updatedHistory: conversationHistory
    };
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Check whether a tool result (or any item in its `results` array) has a
 * `classification` value that appears in the restricted list.
 */
function checkClassification(result, restrictedClassifications) {
  if (!result) return false;

  if (Array.isArray(result.results)) {
    return result.results.some(
      (item) =>
        item.classification && restrictedClassifications.includes(item.classification)
    );
  }

  return (
    result.classification != null &&
    restrictedClassifications.includes(result.classification)
  );
}
