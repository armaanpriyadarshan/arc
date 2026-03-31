/**
 * AgentWall Firewall Evaluation Engine
 *
 * Core decision-making logic that determines whether a tool call should be
 * allowed, blocked, or have specific fields redacted based on active policies.
 */

/**
 * Evaluate a single tool call against the full set of policies.
 *
 * @param {string}  toolName         - The name of the tool being invoked (e.g. "hr_get_employee")
 * @param {object}  toolInput        - The input arguments for the tool call
 * @param {Array}   policies         - All policies from PolicyManager.getAll()
 * @param {boolean} firewallEnabled  - Master on/off switch for the firewall
 * @returns {object} Decision object describing whether the call is allowed
 */
export function evaluateToolCall(toolName, toolInput, policies, firewallEnabled) {
  // Master switch — if the firewall is off, everything is allowed unconditionally.
  if (!firewallEnabled) {
    return { allowed: true };
  }

  // Accumulate field-redaction directives across multiple matching policies.
  let fieldsToRedact = [];
  let firstRedactPolicy = null;

  for (const policy of policies) {
    // Skip disabled policies.
    if (!policy.enabled) continue;

    // Check if this policy applies to the tool being called.
    if (!policy.rules || !policy.rules.tools || !policy.rules.tools.includes(toolName)) {
      continue;
    }

    // --- Determine what kind of policy this is ---

    const hasResponseFieldConditions = policy.rules.conditions?.some(
      (c) => c.field === "response_fields"
    );
    const hasClassifications = Array.isArray(policy.rules.classifications) && policy.rules.classifications.length > 0;
    const hasFilePatterns = Array.isArray(policy.rules.file_patterns) && policy.rules.file_patterns.length > 0;

    // 1. File pattern check (e.g. pol_003 for code_read_file)
    if (hasFilePatterns) {
      const filePath = toolInput?.file_path || "";
      const basename = filePath.split("/").pop();

      const matched = policy.rules.file_patterns.some((pattern) => {
        if (pattern.startsWith("*") && pattern.endsWith("*")) {
          // Wildcard pattern like *secrets* — path must include the inner string
          const inner = pattern.slice(1, -1);
          return filePath.includes(inner);
        }
        // Exact basename match (e.g. ".env", "database.yml")
        return basename === pattern;
      });

      if (matched) {
        return {
          allowed: false,
          reason: policy.reason,
          policyName: policy.name,
          policyId: policy.id
        };
      }
      // If no pattern matched, skip this policy entirely.
      continue;
    }

    // 2. Classification check (e.g. pol_004 for docs tools)
    if (hasClassifications) {
      return {
        allowed: true,
        postCallClassificationCheck: policy.rules.classifications,
        policyName: policy.name,
        policyId: policy.id
      };
    }

    // 3. Field redaction (policy has conditions with field: "response_fields")
    if (hasResponseFieldConditions) {
      for (const condition of policy.rules.conditions) {
        if (condition.field === "response_fields" && Array.isArray(condition.contains)) {
          fieldsToRedact.push(...condition.contains);
        }
      }
      if (!firstRedactPolicy) {
        firstRedactPolicy = { name: policy.name, id: policy.id };
      }
      // Do NOT return yet — keep iterating to merge with other policies that
      // might also want to redact fields on the same tool.
      continue;
    }

    // 4. Full block — no conditions, no classifications, no file_patterns
    return {
      allowed: false,
      reason: policy.reason,
      policyName: policy.name,
      policyId: policy.id
    };
  }

  // If we accumulated any fields to redact, return a redaction decision.
  if (fieldsToRedact.length > 0) {
    const unique = [...new Set(fieldsToRedact)];
    return {
      allowed: true,
      fieldsToRedact: unique,
      policyName: firstRedactPolicy.name,
      policyId: firstRedactPolicy.id
    };
  }

  // No policies matched — allow the call.
  return { allowed: true };
}

const REDACTED = "[REDACTED BY AGENTWALL]";

/**
 * Deep-copy `data` and replace the values of specified fields with a redaction marker.
 *
 * @param {object}        data            - The raw tool result
 * @param {Array<string>} fieldsToRedact  - Dot-notation paths like "payment_method.full_number"
 * @returns {object} A new object with the specified fields replaced
 */
export function redactFields(data, fieldsToRedact) {
  const copy = JSON.parse(JSON.stringify(data));

  function redactSingle(obj, fieldPath) {
    const parts = fieldPath.split(".");
    let current = obj;
    for (let i = 0; i < parts.length - 1; i++) {
      if (current == null || typeof current !== "object") return;
      current = current[parts[i]];
    }
    const lastKey = parts[parts.length - 1];
    if (current != null && typeof current === "object" && lastKey in current) {
      current[lastKey] = REDACTED;
    }
  }

  function applyRedactions(obj) {
    for (const field of fieldsToRedact) {
      redactSingle(obj, field);
    }
  }

  // If the data has a `results` array (e.g. search results), redact each element.
  if (Array.isArray(copy.results)) {
    for (const item of copy.results) {
      applyRedactions(item);
    }
  } else {
    applyRedactions(copy);
  }

  return copy;
}
