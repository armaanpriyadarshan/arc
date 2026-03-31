/**
 * AgentWall Policy Engine
 *
 * Provides default firewall policies and a PolicyManager class
 * for CRUD operations on policies at runtime.
 */

/**
 * Returns the built-in set of six default policies that ship with AgentWall.
 * Each policy targets specific tools and specifies what data to block or redact.
 */
export function getDefaultPolicies() {
  return [
    {
      id: "pol_001",
      name: "Block compensation data",
      description: "Prevent agents from accessing employee compensation data",
      enabled: true,
      effect: "deny",
      reason: "Compensation data requires HR role authorization",
      rules: {
        tools: ["hr_get_compensation", "hr_get_employee"],
        conditions: [
          { field: "response_fields", contains: ["salary", "bonus", "ssn", "equity_shares"] }
        ]
      }
    },
    {
      id: "pol_002",
      name: "Block payment card data",
      description: "Prevent agents from accessing full credit card numbers and CVVs",
      enabled: true,
      effect: "deny",
      reason: "Payment card data requires PCI-DSS authorization",
      rules: {
        tools: ["crm_get_customer"],
        conditions: [
          { field: "response_fields", contains: ["payment_method.full_number", "payment_method.cvv", "payment_method.expiry"] }
        ]
      }
    },
    {
      id: "pol_003",
      name: "Block sensitive files",
      description: "Prevent agents from reading sensitive configuration and secret files",
      enabled: true,
      effect: "deny",
      reason: "Sensitive configuration files require security team authorization",
      rules: {
        tools: ["code_read_file"],
        file_patterns: [".env", "*secrets*", "*credentials*", "database.yml"]
      }
    },
    {
      id: "pol_004",
      name: "Block confidential documents",
      description: "Prevent agents from accessing confidential and restricted documents",
      enabled: true,
      effect: "deny",
      reason: "Document classification level requires elevated access",
      rules: {
        tools: ["docs_get_document", "docs_search"],
        classifications: ["confidential", "restricted"]
      }
    },
    {
      id: "pol_005",
      name: "Block bank account details",
      description: "Prevent agents from accessing company bank account information",
      enabled: true,
      effect: "deny",
      reason: "Bank account data requires Finance role authorization",
      rules: {
        tools: ["finance_get_bank_accounts"]
      }
    },
    {
      id: "pol_006",
      name: "Block PII fields",
      description: "Prevent agents from accessing personally identifiable information",
      enabled: true,
      effect: "deny",
      reason: "PII access requires Data Privacy Officer approval",
      rules: {
        tools: ["hr_get_employee"],
        conditions: [
          { field: "response_fields", contains: ["ssn", "home_address"] }
        ]
      }
    }
  ];
}

/**
 * Manages the lifecycle of firewall policies: list, create, update, delete, toggle.
 * Operates on an in-memory deep copy so the caller's original array is never mutated.
 */
export class PolicyManager {
  constructor(initialPolicies) {
    this.policies = JSON.parse(JSON.stringify(initialPolicies));
    // Track the next numeric suffix for auto-generated IDs.
    // Scan existing policies to find the highest number already in use.
    let maxNum = 0;
    for (const p of this.policies) {
      const match = p.id.match(/^pol_(\d+)$/);
      if (match) {
        const num = parseInt(match[1], 10);
        if (num > maxNum) {
          maxNum = num;
        }
      }
    }
    this._nextNum = maxNum + 1;
  }

  /** Return all policies. */
  getAll() {
    return this.policies;
  }

  /** Return a single policy by its id, or null if not found. */
  getById(id) {
    return this.policies.find((p) => p.id === id) || null;
  }

  /** Create a new policy from the given data, auto-generating a unique id. */
  create(policyData) {
    const id = `pol_${String(this._nextNum).padStart(3, "0")}`;
    this._nextNum++;
    const policy = { id, ...policyData };
    this.policies.push(policy);
    return policy;
  }

  /** Merge `changes` into the policy with the given id. Returns the updated policy, or null. */
  update(id, changes) {
    const policy = this.getById(id);
    if (!policy) return null;
    Object.assign(policy, changes);
    return policy;
  }

  /** Delete a policy by id. Returns true if it existed, false otherwise. */
  delete(id) {
    const idx = this.policies.findIndex((p) => p.id === id);
    if (idx === -1) return false;
    this.policies.splice(idx, 1);
    return true;
  }

  /** Toggle the `enabled` flag on a policy. Returns the updated policy, or null. */
  toggle(id) {
    const policy = this.getById(id);
    if (!policy) return null;
    policy.enabled = !policy.enabled;
    return policy;
  }
}
