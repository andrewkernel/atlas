# ADR 004 — Trusted-code-only MVP

**Status:** accepted.

**Context:** profiling, importing, tracing, installing dependencies and deserializing checkpoints can execute arbitrary code. Strong hostile GPU sandboxing is outside the initial research scope.

**Alternatives:** run any downloaded script; use Docker and call it secure; build a validated hostile-code isolation platform; accept only reviewed registered workloads.

**Evidence:** [Docker security](https://docs.docker.com/engine/security/) documents daemon/configuration attack surfaces; host [resource limits](https://docs.docker.com/engine/containers/resource_constraints/) are not universal GPU isolation. A normal subprocess has the user's file/network permissions.

**Decision:** refuse untrusted workloads in MVP, bound trusted worker lifetime/resources, and keep source/data immutable. Later isolated execution requires explicit approval bound to a concrete execution record and validated controls.

**Tradeoffs:** narrower convenience and weaker native-process isolation; avoids an unsupported safety claim and a separate infrastructure project.

**Revisit when:** a dedicated threat-model review and platform-specific escape/resource/termination tests establish an acceptable isolation boundary. User acknowledgment alone is not evidence that isolation is strong.
